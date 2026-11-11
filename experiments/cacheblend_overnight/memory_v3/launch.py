"""One-shot preflight supervisor; never dispatches remaining items."""
import json,os,signal,subprocess,sys,threading,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import ROOT,save,append,sha
from launch import observe,severe,swap_delta,MIB,GIB
OUT=ROOT/'artifacts/cacheblend_overnight/memory_v3'
def main():
 assert not (OUT/'run-state.json').exists(),'No retries in an existing run directory'
 for name in ['references','results']:(OUT/name).mkdir(exist_ok=True)
 start=time.time();baseline=observe(os.getpid());append(OUT/'memory.jsonl',baseline)
 save(OUT/'run-state.json',{'start_epoch':start,'scope':'single longest frozen-v2 preflight only','code_sha256':{p.name:sha(p) for p in HERE.glob('*.py')},'frozen_sha256':sha(HERE.parent/'review_v2/frozen-v2.json')})
 interrupted=False
 def interrupt(*_):
  nonlocal interrupted
  interrupted=True
 signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
 samples=[baseline];severe_since=None;reason=None;sent=None;caffeine=None
 with (OUT/'worker.log').open('w') as log:
  child=subprocess.Popen([sys.executable,'-u',str(HERE/'worker.py')],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DEACTIVATE_ASYNC_LOAD':'1'})
  timer=threading.Timer(21600,child.kill);timer.daemon=True;timer.start()
  try:
   caffeine=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(child.pid)])
   while child.poll() is None:
    now=observe(child.pid);samples.append(now);append(OUT/'memory.jsonl',now)
    old=next((s for s in samples if s['epoch']>=now['epoch']-60),samples[0])
    if severe(now,old):
     if severe_since is None:severe_since=now['epoch']
    else:severe_since=None
    if reason is None:
     if interrupted:reason='interrupted'
     elif time.time()-start>=21540:reason='deadline_stopped'
     elif severe_since is not None and now['epoch']-severe_since>=30:reason='pressure_stopped'
     elif any(now[k] is None for k in ['os_free_percentage','swapouts_pages','pressure_level']):reason='telemetry_unavailable'
    if reason and sent is None:
     save(OUT/'stop-request.json',{'reason':reason,'elapsed_s':time.time()-start});child.terminate();sent=time.time()
    if sent and time.time()-sent>=30:child.kill();break
    time.sleep(5)
  finally:
   if child.poll() is None:
    child.terminate()
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:child.kill()
   child.wait();timer.cancel()
   if caffeine:
    caffeine.terminate();caffeine.wait()
 now=observe(os.getpid());samples.append(now);append(OUT/'memory.jsonl',now)
 wm=[json.loads(s) for s in (OUT/'worker-memory.jsonl').read_text().splitlines()] if (OUT/'worker-memory.jsonl').exists() else []
 if (OUT/'sampled-worker-memory.jsonl').exists():wm.extend(json.loads(s) for s in (OUT/'sampled-worker-memory.jsonl').read_text().splitlines())
 completed=sorted(p.stem for p in (OUT/'results').glob('*.json'))
 ws=json.loads((OUT/'worker-status.json').read_text()) if (OUT/'worker-status.json').exists() else {}
 delta=swap_delta(baseline,now)
 concerns=[]
 if len(completed)!=5 or ws.get('status')!='complete':concerns.append('All five methods did not finish correctly')
 if any((s['pressure_level'] or 0)>=2 or (s['os_free_percentage'] is not None and s['os_free_percentage']<15) for s in samples):concerns.append('Memory pressure warning/critical or low system free percentage observed')
 if delta is None or delta>128*MIB:concerns.append('New system swapouts exceeded128MiB or unavailable')
 if not wm or max(w['maxrss_bytes'] for w in wm)>12*GIB:concerns.append('Process RSS guard exceeded or unavailable')
 if any(w.get('mps_driver_bytes',0)>.95*w.get('mps_recommended_bytes',1) for w in wm):concerns.append('MPS driver exceeded95% recommended working set')
 result={'status':reason or ('complete' if child.returncode==0 and len(completed)==5 else 'failed'),'returncode':child.returncode,'completed_methods':completed,'elapsed_s':time.time()-start,'worker_status':ws,'process_peak_rss_bytes':max((w['maxrss_bytes'] for w in wm),default=None),'sampled_mps_allocated_peak_bytes':max((w.get('mps_allocated_bytes',0) for w in wm),default=None),'sampled_mps_driver_peak_bytes':max((w.get('mps_driver_bytes',0) for w in wm),default=None),'new_system_swapout_bytes':delta,'new_system_swapin_bytes':max(0,now['swapins_pages']-baseline['swapins_pages'])*now['page_size'] if now['swapins_pages'] is not None and baseline['swapins_pages'] is not None else None,'sufficient_headroom_observed':not concerns,'headroom_concerns':concerns,'remaining_batch_authorized':False,'memory_limits':'MPS peaks are sampled; RSS/MPS overlap. Swap counters are system-wide cumulative traffic.'}
 save(OUT/'status.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
