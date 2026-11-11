"""Bounded supervisor: one child inference job, OS telemetry, no automatic retries."""
import argparse,json,os,re,signal,subprocess,sys,time,threading
from common import HERE,ROOT,OUT,FROZEN,sha,save,append
GIB=1024**3;MIB=1024**2

def command(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=4)
 return p.stdout.strip() if p.returncode==0 else None

def observe(pid):
 vm=command(['/usr/bin/vm_stat']) or ''
 page=re.search(r'page size of (\d+)',vm)
 size=int(page.group(1)) if page else None
 def counter(name):
  m=re.search(re.escape(name)+r':\s+(\d+)',vm)
  return int(m.group(1)) if m else None
 p=command(['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level'])
 free=command(['/usr/bin/memory_pressure','-Q']) or ''
 m=re.search(r'free percentage:\s*(\d+)%',free)
 rss=command(['/bin/ps','-o','rss=','-p',str(pid)])
 return {'epoch':time.time(),'pressure_level':int(p) if p and p.isdigit() else None,
  'os_free_percentage':int(m.group(1)) if m else None,'page_size':size,
  'swapouts_pages':counter('Swapouts'),'swapins_pages':counter('Swapins'),
  'compressor_pages':counter('Pages occupied by compressor'),
  'free_pages':counter('Pages free'),'inactive_pages':counter('Pages inactive'),
  'swapusage':command(['/usr/sbin/sysctl','-n','vm.swapusage']),
  'process_rss_bytes':int(rss)*1024 if rss and rss.strip().isdigit() else None}

def swap_delta(a,b):
 if a['swapouts_pages'] is None or b['swapouts_pages'] is None or not b['page_size']:return None
 return max(0,b['swapouts_pages']-a['swapouts_pages'])*b['page_size']

def severe(now,old):
 delta=swap_delta(old,now)
 return ((now['pressure_level'] or 0)>=4 or
  (now['os_free_percentage'] is not None and now['os_free_percentage']<=10) or
  (delta is not None and delta>256*MIB and ((now['pressure_level'] or 0)>=2 or (now['os_free_percentage'] is not None and now['os_free_percentage']<=20))))

def comfortable(samples,wm,pre,total_items):
 reasons=[]
 if not samples or any(s['os_free_percentage'] is None or s['swapouts_pages'] is None for s in samples):reasons.append('Required OS free/swap telemetry unavailable')
 if any((s['pressure_level'] or 0)>=2 or (s['os_free_percentage'] is not None and s['os_free_percentage']<15) for s in samples):reasons.append('Preflight showed memory warning or free percentage below15')
 delta=swap_delta(samples[0],samples[-1]) if samples else None
 if delta is None or delta>128*MIB:reasons.append('Preflight swapout growth exceeds128MiB or is unavailable')
 if not wm or max(w['maxrss_bytes'] for w in wm)>12*GIB:reasons.append('Process peak RSS exceeds12GiB or metrics unavailable')
 if any(w.get('mps_driver_bytes',0)>.95*w.get('mps_recommended_bytes',1) for w in wm):reasons.append('Observed MPS driver memory exceeds95% recommended working set')
 estimate=pre['elapsed_s_including_load']+1.5*pre['item_wall_s_including_validation']*(total_items-1)
 if estimate>19800:reasons.append('Conservative estimated runtime exceeds5.5hours')
 return {'continue':not reasons,'reasons':reasons,'estimated_total_s_conservative':estimate,'preflight_swapout_increase_bytes':delta,'samples':len(samples),'observed_process_maxrss_bytes':max((w['maxrss_bytes'] for w in wm),default=None),'observed_mps_driver_max_bytes':max((w.get('mps_driver_bytes',0) for w in wm),default=None)}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
 for name in ['results','references']:(OUT/name).mkdir(parents=True,exist_ok=True)
 phash=sha(FROZEN);protocol=json.loads(FROZEN.read_text())
 codepaths=[HERE/x for x in ['common.py','worker.py','launch.py','report.py']]
 hashes={str(p.relative_to(ROOT)):sha(p) for p in codepaths}
 path=OUT/'run-state.json'
 if path.exists():
  state=json.loads(path.read_text())
  assert args.resume,'Existing run: explicit --resume required; no automatic retry'
  assert state['protocol_sha256']==phash and state['code_sha256']==hashes,'Protocol/code changed; resume refused'
  status=json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {}
  assert status.get('status') not in ('complete','failed','preflight_stopped','deadline_stopped','pressure_stopped'),'Stopped/final run is not automatically retried'
 else:
  assert not args.resume
  state={'start_epoch':time.time(),'protocol_sha256':phash,'code_sha256':hashes,'six_hour_limit_s':21600}
  save(path,state)
 interrupted=False
 def on_signal(*_):
  nonlocal interrupted
  interrupted=True
 signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
 child=None;samples=[];severe_since=None;stop_reason=None;sent=None
 try:
  with (OUT/'worker.log').open('a') as log:
   child=subprocess.Popen([sys.executable,'-u',str(HERE/'worker.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
   timer=threading.Timer(max(0,state['start_epoch']+21600-time.time()),lambda: child.kill() if child.poll() is None else None);timer.daemon=True;timer.start()
   save(OUT/'status.json',{'status':'running_preflight','pid':child.pid,**state})
   while child.poll() is None:
    now=observe(child.pid);samples.append(now);append(OUT/'memory.jsonl',now)
    elapsed=time.time()-state['start_epoch']
    old=next((s for s in samples if s['epoch']>=now['epoch']-60),samples[0])
    if severe(now,old):
     if severe_since is None:severe_since=now['epoch']
    else:severe_since=None
    if stop_reason is None:
     if interrupted:stop_reason='interrupted'
     elif elapsed>=21540:stop_reason='deadline_stopped'
     elif severe_since is not None and now['epoch']-severe_since>=30:stop_reason='pressure_stopped'
     elif now['os_free_percentage'] is None or now['swapouts_pages'] is None:stop_reason='failed: required telemetry unavailable'
    if (OUT/'preflight.json').exists() and not (OUT/'decision.json').exists():
     pre=json.loads((OUT/'preflight.json').read_text());wm=[json.loads(s) for s in (OUT/'worker-memory.jsonl').read_text().splitlines()]
     decision=comfortable(samples,wm,pre,len(protocol['items']));save(OUT/'decision.json',decision)
     if not decision['continue']:stop_reason='preflight_stopped'
     else:save(OUT/'status.json',{'status':'running_batch','elapsed_s':elapsed,'preflight':decision,'pid':child.pid})
     print('PREFLIGHT',json.dumps(decision),flush=True)
    if stop_reason and sent is None:
     save(OUT/'stop-request.json',{'reason':stop_reason,'elapsed_s':elapsed});child.terminate();sent=time.time()
    if elapsed>=21600 or (sent is not None and time.time()-sent>=30):
     child.kill();break
    time.sleep(min(10,max(.1,21600-elapsed)))
 except BaseException as e:
  stop_reason='failed'
  save(OUT/'supervisor-error.json',{'error':repr(e),'elapsed_s':time.time()-state['start_epoch']})
 finally:
  if child and child.poll() is None:
   child.terminate()
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:child.kill();child.wait()
 if child:
  if 'timer' in locals():timer.cancel()
  ws=json.loads((OUT/'worker-status.json').read_text()) if (OUT/'worker-status.json').exists() else {}
  result={'status':stop_reason or ('complete' if child.returncode==0 and ws.get('status')=='complete' else 'failed'),'returncode':child.returncode,'elapsed_s':time.time()-state['start_epoch'],'completed_results':len(list((OUT/'results').glob('*.json'))),'expected_results':5*len(protocol['items']),'worker_status':ws,'stop_reason':stop_reason}
  save(OUT/'status.json',result)
  subprocess.run([sys.executable,str(HERE/'report.py')],cwd=ROOT,check=True)
  print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
