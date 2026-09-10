"""Per-defect F1 at a fixed decision threshold p >= 0.50 (no in-sample tuning).
Also reports the in-sample balanced-accuracy-optimised F1 for reference.
Usage: python defect_f1_fixed.py  (run from repo root)"""
import json, glob, csv, numpy as np
DEFECTS = ["bruising","discoloration","leaking","mold","visible_cut","wrinkling"]
MODELS = {"llava15_7b":"LLaVA-1.5 7B","qwen2_vl_7b":"Qwen2-VL 7B","qwen25_vl_7b":"Qwen2.5-VL 7B",
          "internvl3_8b":"InternVL3 8B","llama32_11b_vision":"Llama-3.2 11B-V"}
TAU = 0.50

def prf(pred, y):
    tp=int((pred&y).sum()); fp=int((pred&~y).sum()); fn=int((~pred&y).sum())
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f = 2*p*r/(p+r) if p+r else 0.0
    return p,r,f,tp,fp,fn

rows=[]
for slug,name in MODELS.items():
    recs=[json.loads(l) for l in open(f"results/raw/{slug}__defects__logit_per_defect.jsonl")]
    recs=[r for r in recs if r.get("defect_scores") and r.get("gt")]   # non-fresh, >=1 annotated defect
    n=len(recs)
    for d in DEFECTS:
        s=np.array([r["defect_scores"][d] for r in recs]); y=np.array([d in r["gt"] for r in recs])
        p,r,f,tp,fp,fn=prf(s>=TAU,y)
        # in-sample reference: tau maximising balanced accuracy
        best=(-1,None)
        if y.sum()==0:
            rows.append(dict(model=name,defect=d,n=n,n_pos=0,tau=TAU,precision=None,recall=None,f1_fixed=None,
                             tp=tp,fp=fp,fn=fn,n_pred_pos=int((s>=TAU).sum()),tau_insample=None,f1_insample=None)); continue
        for t in np.round(np.arange(0.05,1.0,0.05),2):
            pr=s>=t; sens=(pr&y).sum()/y.sum(); spec=(~pr&~y).sum()/(~y).sum() if (~y).sum() else 1
            ba=(sens+spec)/2
            if ba>best[0]: best=(ba,t)
        _,_,f_opt,*_=prf(s>=best[1],y)
        rows.append(dict(model=name,defect=d,n=n,n_pos=int(y.sum()),tau=TAU,
                         precision=round(p,3),recall=round(r,3),f1_fixed=round(f,3),
                         tp=tp,fp=fp,fn=fn,n_pred_pos=int((s>=TAU).sum()),
                         tau_insample=best[1],f1_insample=round(f_opt,3)))
with open("defect_f1_fixed.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

