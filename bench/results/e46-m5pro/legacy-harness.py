"""Shared-weight, shuffled kernel comparisons with a duplicate control and raw evidence.

Run only through the guarded lab runner. Does not stop or restore any service itself.
"""
import argparse,hashlib,importlib,json,random,statistics,time
from pathlib import Path
import numpy as np
import mlx.core as mx
from tokenizers import Tokenizer
from strata_mlx import engine as eng, fused, blocks, attention
from strata_mlx.mtp import load_mtp
from strata_mlx.sampler import Sampling
loadmod=importlib.import_module('strata_mlx.load')

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--mtp',required=True);p.add_argument('--tokenizer',required=True);p.add_argument('--fixtures',required=True);p.add_argument('--output',required=True);p.add_argument('--micro-rounds',type=int,default=20);p.add_argument('--answer-rounds',type=int,default=3)
a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=True)
def save(name,data): (out/name).write_text(json.dumps(data,indent=2)+'\n')
def append(name,data):
 with (out/name).open('a') as f:f.write(json.dumps(data)+'\n')
def phase(name):append('phase-events.jsonl',{'phase':name,'monotonic':time.monotonic(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())});print(name,flush=True)
def clean(text):return text.replace('<|im_end|>','').replace('<|endoftext|>','').split('</think>')[-1].strip()
flags=[(fused,'ROUTER_TOPK',True),(fused,'ROUTER_AS_IS',True),(fused,'COMBINE_UPTO',16),(fused,'GDN_PREP_UPTO',16),(fused,'GATED_NORM_UPTO',16),(blocks,'GLU_IN_DOWN',True),(blocks,'BLOCK_GLU',True),(attention,'QK_ROPE_UPTO',16)]
def configure(enabled):
 for mod,key,value in flags:setattr(mod,key,value if enabled else False)
# A switch-path reference, not a historical-revision reproduction: current draft-head and thread rules shared.
settings={'reference6':(False,6),'new6':(True,6),'new8':(True,8),'control6':(True,6)}
tok=Tokenizer.from_file(a.tokenizer);encode=lambda s:tok.encode(s,add_special_tokens=False).ids
stop=(248044,248046)
phase('load');t=time.monotonic();model,meta=loadmod.load(a.model,mx.float16,expert_memory='auto')
load_s=time.monotonic()-t
threads=getattr(model,'table_warming',None)
info={'device':mx.device_info(),'auto_fuse_upto':eng.FUSE_UPTO,'model':a.model,'load_s':load_s,'table_warming_started':bool(threads),'active_bytes_after_load':mx.get_active_memory(),'cache_bytes_after_load':mx.get_cache_memory(),'settings':settings,'precision':'float16; MLX_ENABLE_TF32 unset','kernel_reference_limit':'Reference toggles eight trunk module switches and fuse limit; current shared draft-head and Q2 thread rule remain. Not an exact historical source comparison.','seed':20261015,'full_model_resident':model.expert_store is None}
phase('draft-load');t=time.monotonic();mtp=load_mtp(a.mtp,model.args.text,mx.float16);info['draft_load_s']=time.monotonic()-t
if model.expert_store is not None:
 info['expert_budget_bytes']=model.expert_store.fit(64*2**30)
# Cap temporary MLX recycling independently of weight residency.
mx.set_cache_limit(512*2**20)
engines={}
for label,(enabled,limit) in settings.items():
 phase('warm-'+label);configure(enabled);e=eng.Engine(model,mtp,fuse_upto=limit);t=time.monotonic();e.warm();e.generate(list(range(1000,1024)),12,sampling=Sampling(temperature=0,seed=42));e.reset();engines[label]=e;info.setdefault('warm_s',{})[label]=time.monotonic()-t
if threads:
 phase('wait-table-warming')
 for thread in threads:
  if thread:thread.join(timeout=180)
 info['table_warming_finished']=all(not thread or not thread.is_alive() for thread in threads)
 if not info['table_warming_finished']:raise RuntimeError('Table warm exceeded bound')
else:info['table_warming_finished']=False
save('configuration.json',info)
# No overlapping model executions. Engines share weights, own small recurrent/KV state.
rng=random.Random(20261015)
prefix=encode('<|im_start|>user\nReturn exactly the word READY.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
phase('micro')
for window in [1,4,7,8]:
 ids=mx.array([[1000+i for i in range(window)]],dtype=mx.int32)
 for label,e in engines.items():
  configure(settings[label][0]);e.reset();e.read(prefix)
  logits,w=e._forward(ids);mx.eval(logits,*eng._arrays([list(w.gdn),w.ple_conv,w.residual]))
  for cache in e.attn.values():cache.trim(window)
 for repetition in range(a.micro_rounds):
  order=list(settings);rng.shuffle(order)
  for position,label in enumerate(order):
   configure(settings[label][0]);e=engines[label];mx.synchronize();t=time.perf_counter()
   logits,w=e._forward(ids);mx.eval(logits,*eng._arrays([list(w.gdn),w.ple_conv,w.residual]));elapsed=(time.perf_counter()-t)*1000
   raw=np.array(logits)
   append('micro.jsonl',{'round':repetition,'window':window,'position':position,'setting':label,'ms':elapsed,'logits_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),'argmax':raw.argmax(-1).tolist()})
   for cache in e.attn.values():cache.trim(window)
   del logits,w,raw
# Model-state workspaces reset before each answer, including draft row predictor cache.
fixtures=json.loads(Path(a.fixtures).read_text())
phase('answers')
for repetition in range(a.answer_rounds):
 for case in fixtures:
  order=list(settings);rng.shuffle(order)
  for position,label in enumerate(order):
   configure(settings[label][0]);e=engines[label];e.reset();e.rows=eng.GuessRows(model.lm_head,model.model.args.vocab_size);e.sampling=Sampling(temperature=0,seed=42)
   ids=encode(case['prompt']);start=time.perf_counter();tokens=[e.read(ids)];ttft=time.perf_counter()-start;visible=None
   if clean(tok.decode(tokens,skip_special_tokens=False)):visible=ttft
   while len(tokens)<256 and tokens[-1] not in stop:
    tokens+=e.step(256-len(tokens),stop)
    if visible is None and clean(tok.decode(tokens,skip_special_tokens=False)):visible=time.perf_counter()-start
   wall=time.perf_counter()-start;text=clean(tok.decode(tokens,skip_special_tokens=False))
   try:correct=json.loads(text)==case['expected'] if 'expected' in case else None
   except (ValueError,TypeError):correct=False
   append('answers.jsonl',{'round':repetition,'case':case['name'],'position':position,'setting':label,'prompt_tokens':len(ids),'prompt_ids':ids,'output_ids':tokens,'text':text,'correct':correct,'expected':case.get('expected'),'ttft_s':ttft,'first_visible_s':visible,'wall_s':wall,'decode_tps':(len(tokens)-1)/(wall-ttft) if wall>ttft else None,'finish':'stop' if tokens[-1] in stop else 'length','stats':vars(e.stats),'expert_metrics':{k:getattr(model.expert_store,k) for k in ['hits','misses','bytes_read','read_s','waited_s']} if model.expert_store else {}})
   phase(f'answer-{repetition}-{case["name"]}-{label}')
phase('complete')
