"""Roadmap #47: where the committee screen loses the true triple.

For one three-fault case, rank every pair the way #33 does, extend the best 200 pairs to triples, fit
each triple continuously and order them as the committee does. Report where the true triple's sub-pairs
rank, whether the true triple is among the extended triples at all, and if so where it ranks and how
many triples fit the practice labels at least as well.

    python experiments/locate-repair/screen_ranks.py 49 runs/screen-misses
    python experiments/locate-repair/screen_ranks.py 126 runs/unseen-misses 1000   # extend 1,000 pairs (roadmap #55)
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("early_values", HERE/"early_values.py"); ev=importlib.util.module_from_spec(spec); sys.modules["early_values"]=ev; spec.loader.exec_module(ev)
st,tc,t3,lr=ev.st,ev.tc,ev.t3,ev.lr
sc,js,ds,ct,tr=tc.sc,tc.js,tc.ds,tc.ct,tc.tr
seed=int(sys.argv[1])
limit=int(sys.argv[3]) if len(sys.argv)>3 else t3.PAIRS
case=ev.rs.resize(t3.corrupt(st.WIDTH, seed)); px,py,_=lr.split(case["practice"])
faults=case["faults"]; truth=tuple(sorted(tuple(f[:2]) for f in faults))
pairs=sc.ranked_pairs(case["arrays"],px,py)
ranked=[pairs[k] for k in js.joint_order(js.joint_scores(case,px,py,pairs,points=ds.PAIR_POINTS))]
def isub(p): return all(tuple(e) in truth for e in p)
sub_ranks=[k for k,p in enumerate(ranked) if isub(p)]
raw_ranks=[k for k,p in enumerate(pairs) if isub(p)]
triples=t3.triples_from(ranked, lr.entries(case["arrays"]), limit)
in_triples=truth in triples
coarse=[t3.fit_triple(case,px,py,t,7) for t in triples]
acc7,cost7=np.array([b[0][0] for b in coarse]),np.array([-b[0][1] for b in coarse])
rank7=np.empty(len(triples),int); rank7[tr.order(acc7,cost7,None)]=np.arange(len(triples))
fits=[ct.best_of(*ct.continuous_fits(case,px,py,t,c)) for t,c in zip(triples,coarse)]
acc,cost=np.array([b[0][0] for b in fits]),np.array([-b[0][1] for b in fits])
order=list(tr.order(acc,cost,rank7))
told=st.told_start(case,px,py,tuple(tuple(f[:2]) for f in faults))
out=dict(seed=seed,faults=faults,sizes=case["sizes"],pairs=len(pairs),limit=limit,
         true_subpair_ranks_joint=sub_ranks,true_subpair_ranks_raw=raw_ranks,triples=len(triples),in_triples=in_triples,
         told_acc=told[0][0],told_cost=-told[0][1],acc_200th=float(acc[order[199]]),acc_best=float(acc[order[0]]),
         n_at_or_above_told=int((acc>told[0][0]).sum()+((acc==told[0][0])&(cost<=-told[0][1])).sum()))
if in_triples:
    k=triples.index(truth); out.update(rank_continuous=order.index(k),rank_coarse7=int(rank7[k]),acc_in=float(acc[k]),acc7_in=float(acc7[k]))
Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
print(json.dumps(out)); json.dump(out, open(Path(sys.argv[2]) / (f"rank-{seed}.json" if limit==t3.PAIRS else f"rank-{seed}-{limit}.json"), "w"))
