"""Capture ONE real INTACT-RA-Agentic decision, end to end, for the worked example.  [AUTO]

Scenario S16_high_ceiling, development seed 20260925: the first scored epoch
(>= 110) in which the agent executes more than one write.  Writes
results/S16_high_ceiling/worked_example/trace.json with every value used in
docs/WORKED_EXAMPLE.md, including the full Kalman state at decision time."""
import sys, json, copy, numpy as np; sys.path.insert(0,'.')
from intact_agentic.config import load_config, build_registry
from intact_agentic.experiment import Experiment
from intact_agentic.arbiter.sensitivity import StaticSensitivity
from intact_agentic.arbiter.margins import margin
from intact_agentic import methods as M
SC="S16_high_ceiling"; c=load_config(scenario=SC); reg=build_registry(c)
prior=StaticSensitivity.load(c,f"artifacts/prior_{SC}.json")
c["run"].update({"epochs":400,"log_every":0,"checkpoint_every":0,"counterfactual":False})
ex=Experiment(c,M.get("intact-ra-agentic"),reg,"/tmp/trace",seed=20260925,prior=prior,telemetry=False,log=lambda *a:None)
A=ex.arbiter; cap={}
orig_decide=A.decide
def dec(**kw):
    out=orig_decide(**kw); decisions,win,info=out
    if ex.epoch>=110 and len([d for d in decisions if d.executed])>=2 and "trace" not in cap:
        S=kw["sens"]; R=kw["regimes"]; props=kw["proposals"]; cl=kw["claims"]; ints=kw["intents"]
        reqs=A.last_reqs
        cap["trace"]=dict(epoch=ex.epoch, controls={k:v for k,v in kw["controls"].items() if k.startswith("quota")},
          proposals=props, g_now=kw["g_now"],
          regimes={i:R.of_intent(i) for i in ints},
          reqs={j:dict(nu_old=r.nu_old,nu_req=r.nu_req,nu_feasible=r.nu_feasible,dose=r.dose,trust=r.clipped_by_trust,env=r.clipped_by_envelope) for j,r in reqs.items()},
          slopes={f"{cl[j].param}|{i}":[S.get(R.of_intent(i),cl[j].param,i),S.sigma(R.of_intent(i),cl[j].param,i)] for j in props for i in ints},
          winner=dict(claims=list(win.claims),utility=win.utility,dg=win.dg,sigma=win.sigma,g_hat=win.g_hat,g_low=win.g_lower,
                      adjusted={j:dict(nu=r.nu_feasible,dose=r.dose) for j,r in win.reqs.items()}),
          top=[dict(claims=list(s.claims),utility=s.utility,admissible=s.admissible,reason=s.reject_reason) for s in sorted(A.last_scores,key=lambda s:-s.utility)[:8]],
          decisions=[dict(jid=d.jid,param=d.param,nu_old=d.nu_old,nu_star=d.nu_star,outcome=d.outcome.value,executed=d.executed) for d in decisions],
          beta=A.beta, eps={i:ints[i].epsilon for i in ints},
          kalman={i:(lambda key: dict(params=S.params, theta=S._kth[key].tolist(),
                                      C=S._kC[key].tolist(), R=S._noise.get(i)))(S._kslot(R.of_intent(i),i))
                  for i in ints} if hasattr(S,"_kth") else {},
          intents={i:dict(tenant=ints[i].tenant,kpi=ints[i].kpi,target=ints[i].target) for i in ints})
        cap["pending"]=True
    return out
A.decide=dec
rows={}
for ep in range(400):
    pre=None
    if cap.get("pending") is None: pass
    kpm_hold={}
    ex.step_epoch()
    if cap.get("pending"):
        rec=ex.metrics.records[-1]
        cap["trace"]["g_before"]=rec.g_before; cap["trace"]["g_after"]=rec.g_after; cap["trace"]["predicted"]=rec.predicted
        cap["trace"]["kpm_cell"]={k:rec.kpm_cell.get(k) for k in ("prb_util_pct","delivered_mbps","offered_mbps")}
        last=ex.sens.residuals[-12:] if ex.sens.residuals else []
        cap["trace"]["kalman_updates"]=[{k:(v if not isinstance(v,float) else round(v,5)) for k,v in r.items() if k!="x"} for r in last if r["epoch"]==cap["trace"]["epoch"]]
        cap["trace"]["noise_var"]={i:ex.sens._noise.get(i) for i in reg.intents}
        cap["trace"]["telemetry"]={t:{k:ex.ran.slices[t] and None for k in ()} for t in ex.ran.slices}
        cap["pending"]=False; break
import os; os.makedirs("results/S16_high_ceiling/worked_example", exist_ok=True)
json.dump(cap.get("trace",{}),open("results/S16_high_ceiling/worked_example/trace.json","w"),indent=1,default=float)
print("captured epoch", cap.get("trace",{}).get("epoch"), "winner", cap.get("trace",{}).get("winner",{}).get("claims"))
