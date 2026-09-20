"""Per-defect precision / recall / F1 at a fixed decision threshold p >= 0.50.
No threshold tuning of any kind. Defects with no annotated positives
(leaking) are written with empty metrics and should be left out of comparisons.
Usage: python defect_f1_fixed_default.py  (run from repo root)"""
import json, csv, numpy as np
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
    recs=[json.loads(l) for l in open(f"results/raw/{slug}__defects__logit_per_defect.jsonl") if l.strip()]
    recs=[r for r in recs if r.get("defect_scores") and r.get("gt")]   # non-fresh, >=1 annotated defect
    n=len(recs); f1s=[]
    for d in DEFECTS:
        s=np.array([r["defect_scores"][d] for r in recs]); y=np.array([d in r["gt"] for r in recs])
        p,r,f,tp,fp,fn=prf(s>=TAU,y)
        if y.sum()==0:
            rows.append(dict(model=name,defect=d,n=n,n_pos=0,tau=TAU,precision=None,recall=None,f1=None,
                             tp=tp,fp=fp,fn=fn,n_pred_pos=int((s>=TAU).sum()))); continue
        f1s.append(f)
        rows.append(dict(model=name,defect=d,n=n,n_pos=int(y.sum()),tau=TAU,
                         precision=round(p,3),recall=round(r,3),f1=round(f,3),
                         tp=tp,fp=fp,fn=fn,n_pred_pos=int((s>=TAU).sum())))
    rows.append(dict(model=name,defect="macro",n=n,n_pos=None,tau=TAU,precision=None,recall=None,
                     f1=round(float(np.mean(f1s)),3) if f1s else None,tp=None,fp=None,fn=None,n_pred_pos=None))
with open("defect_f1_fixed.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)