import sys, json, random, pathlib, argparse
import numpy as np, torch
from world import *
from train import Net
ROOT=pathlib.Path(__file__).resolve().parent.parent

def load():
 ck=torch.load(ROOT/'artifacts/world_dynamics.pt',map_location='cpu',weights_only=True);m=Net();m.load_state_dict(ck['state_dict']);m.eval();return m

def predict(m,s,a):
 with torch.no_grad(): z=m(torch.tensor([encode(s,a)],dtype=torch.float32))[0].numpy()
 for i,k in enumerate(FIELDS):
  if k in BOOL:z[i]=1/(1+np.exp(-np.clip(z[i],-50,50)))
 return decode(z.astype(float))

def sequence(m,steps,count,seed):
 rng=random.Random(seed); maes=[];bools=[];path=[];fin=[]
 for ep in range(count):
  truth=fresh();pred=fresh(); em=[];ba=[]
  # mix random actions and purposeful sequences, but sample paths are reproducible
  seq=[rng.choice(ACTIONS) for _ in range(steps)]
  if ep%3==0:
   seq[:min(5,steps)]=['seed','rain','sun','dam','open'][:min(5,steps)]
  for a in seq:
   truth=transition(truth,a);pred=predict(m,pred,a)
   em.append(np.mean([abs(truth[k]-pred[k]) for k in ['wet','eco']]))
   ba.append(np.mean([truth[k]==pred[k] for k in BOOL]))
  maes.append(em);bools.append(ba);fin.append({'episode':ep,'actions':seq,'truth':truth,'pred':pred,'final_mae':float(em[-1])})
 a=np.asarray(maes);b=np.asarray(bools)
 return {'steps':steps,'episodes':count,'mean_mae_by_step':np.round(a.mean(0),3).tolist(),'mean_bool_accuracy_by_step':np.round(b.mean(0),4).tolist(),'final_mae':round(float(a[:,-1].mean()),3),'final_bool_accuracy':round(float(b[:,-1].mean()),4),'overall_mae':round(float(a.mean()),3),'final_mae_p90':round(float(np.percentile(a[:,-1],90)),3),'examples':sorted(fin,key=lambda x:-x['final_mae'])[:3]}

def boundary(m):
 states=[]
 base=fresh()
 for wet in [0,1,8,9,10,11,35,36,44,44.9,45,45.1,46,65,90,99,100]:
  for eco in [0,10,40,70,100]:
   for seed in [False,True]:
    for sun in [False,True]:
     s=dict(base,wet=wet,eco=eco,seed=seed,sun=sun,rain=not sun)
     states.append(s)
 for dam in [False,True]:
  for grown in [False,True]:
   for gate in [False,True]:
    for lit in [False,True]:
     states.append(dict(base,wet=45,eco=70,seed=True,sun=True,dam=dam,grown=grown,gate=gate,lit=lit))
 errors=[]
 for s in states:
  for act in ACTIONS:
   t=transition(s,act);p=predict(m,s,act)
   wrong=[k for k in BOOL if t[k]!=p[k]]
   mae=float(np.mean([abs(t[k]-p[k]) for k in ['wet','eco']]))
   errors.append({'state':s,'action':act,'truth':t,'pred':p,'bool_failures':wrong,'numeric_mae':round(mae,3)})
 return {'cases':len(errors),'bool_accuracy':round(1-sum(len(e['bool_failures']) for e in errors)/(len(errors)*len(BOOL)),4),'numeric_mae':round(float(np.mean([e['numeric_mae'] for e in errors])),3),'any_boolean_failure_rate':round(float(np.mean([bool(e['bool_failures']) for e in errors])),4),'worst_cases':sorted(errors,key=lambda e:(len(e['bool_failures']),e['numeric_mae']),reverse=True)[:30],'important_note':'Includes deliberately inconsistent/unreachable states to probe generalization; report these separately from reachable rollouts.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=100);args=p.parse_args();torch.set_num_threads(2);m=load()
 out={'rollout10':sequence(m,10,args.episodes,2401),'rollout50':sequence(m,50,args.episodes,2402),'boundary':boundary(m)}
 (ROOT/'artifacts/long_horizon_metrics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print({k:{q:v for q,v in x.items() if q in ['final_mae','final_bool_accuracy','numeric_mae','bool_accuracy','cases','final_mae_p90']} for k,x in out.items()})
if __name__=='__main__':main()
