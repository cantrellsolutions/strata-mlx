"""Paired bootstrap intervals and duplicate-control noise; never promote on a headline rate alone."""
import argparse,json,random,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('results');a=p.parse_args();root=Path(a.results)
def read(p):return [json.loads(x) for x in p.read_text().splitlines()]
def interval(values):
 rng=random.Random(20261015);boots=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(2000));return [boots[49],boots[1949]]
summary={'method':'Paired round medians, seeded 2000-resample bootstrap interval. Baseline is new6; control6 repeats the same setting. Positive savings means faster. Timing evidence only; parity/quality gate is separate.','packs':{}}
for pack in ['Q2_0','IQ3_S']:
 r=root/pack
 if not (r/'micro.jsonl').exists():continue
 rows=read(r/'micro.jsonl');answers=read(r/'answers.jsonl') if (r/'answers.jsonl').exists() else []
 out={'configuration':json.loads((r/'configuration.json').read_text()),'micro':[],'answers':{}}
 for window in [1,4,7,8]:
  grouped={}
  for x in rows:
   if x['window']==window:grouped.setdefault(x['round'],{})[x['setting']]=x
  complete=[d for d in grouped.values() if set(d)=={'reference6','new6','new8','control6'}]
  noise=[d['new6']['ms']-d['control6']['ms'] for d in complete]
  if not complete:continue
  for setting in ['reference6','new8','control6']:
   savings=[d['new6']['ms']-d[setting]['ms'] for d in complete];ci=interval(savings)
   out['micro'].append({'window':window,'setting':setting,'paired_rounds':len(complete),'median_ms':statistics.median(d[setting]['ms'] for d in complete),'new6_median_ms':statistics.median(d['new6']['ms'] for d in complete),'median_saving_ms':statistics.median(savings),'bootstrap_95_median_interval_ms':ci,'duplicate_control_median_ms':statistics.median(noise),'duplicate_control_abs_p95_ms':sorted(map(abs,noise))[int(.95*(len(noise)-1))],'logits_bit_equal_to_new6':all(d['new6']['logits_sha256']==d[setting]['logits_sha256'] for d in complete),'argmax_equal_to_new6':all(d['new6']['argmax']==d[setting]['argmax'] for d in complete)})
 for setting in ['reference6','new6','new8','control6']:
  x=[d for d in answers if d['setting']==setting];graded=[d for d in x if d['correct'] is not None]
  out['answers'][setting]={'requests':len(x),'graded':len(graded),'correct':sum(d['correct'] for d in graded),'completion_limits':sum(d['finish']=='length' for d in x),'per_case':{case:{'wall_s':[d['wall_s'] for d in x if d['case']==case],'decode_tps':[d['decode_tps'] for d in x if d['case']==case],'ttft_s':[d['ttft_s'] for d in x if d['case']==case]} for case in sorted({d['case'] for d in x})}}
 paired={}
 for x in answers:paired.setdefault((x['round'],x['case']),{})[x['setting']]=x
 out['answer_parity']=[{'round':k[0],'case':k[1],'setting':setting,'output_ids_equal_to_new6':d[setting]['output_ids']==d['new6']['output_ids'],'visible_text_equal_to_new6':d[setting]['text']==d['new6']['text']} for k,d in paired.items() for setting in ['reference6','new8','control6'] if setting in d and 'new6' in d]
 summary['packs'][pack]=out
(root/'comparison-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:{'micro_rows':len(v['micro']),'answers':{s:x['requests'] for s,x in v['answers'].items()}} for k,v in summary['packs'].items()}))
