import importlib.util,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/cacheblend_overnight'))
from common import parse,score,overlap
from launch import comfortable,severe,swap_delta,MIB,GIB

class Tokenizer:
 def encode(self,s,add_special_tokens=False):return s.split()

def test_reference_scoring_keeps_real_first_token_and_handles_empty():
 s=score('Paris',['Rome'],Tokenizer())
 assert s['reference_qwen_token_f1']==0
 assert s['legacy_drop_first_token_f1_DIAGNOSTIC_ONLY']==1
 assert score('',['Paris'],Tokenizer())['reference_qwen_token_f1']==0
 assert parse('\nYes, indeed\nextra')=='Yes'
 assert score('The answer is Paris.',['Paris'],Tokenizer())['existing_em']==0

def test_pressure_and_comfort_reject_memory_risk():
 a={'epoch':0,'pressure_level':1,'os_free_percentage':50,'page_size':16384,'swapouts_pages':10}
 b={**a,'epoch':60,'swapouts_pages':10+20000,'pressure_level':2}
 assert severe(b,a)
 wm=[{'maxrss_bytes':10*GIB,'mps_driver_bytes':9*GIB,'mps_recommended_bytes':12*GIB}]
 pre={'elapsed_s_including_load':60,'item_wall_s_including_validation':50}
 assert comfortable([a,a],wm,pre,10)['continue']
 assert not comfortable([a,b],wm,pre,10)['continue']
 assert not comfortable([a,a],[{**wm[0],'maxrss_bytes':13*GIB}],pre,10)['continue']
