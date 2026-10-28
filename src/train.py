import argparse, json, pathlib, random, sys
import numpy as np
import torch
from torch import nn
from world import ACTIONS,FIELDS,BOOL,fresh,transition,sample_episodes,random_reachable,encode,target,decode
ROOT=pathlib.Path(__file__).resolve().parent.parent
class Net(nn.Module):
    def __init__(self,h=64):
        super().__init__();self.net=nn.Sequential(nn.Linear(16,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,10))
    def forward(self,x):return self.net(x)

def main():
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=1400);p.add_argument('--epochs',type=int,default=55);p.add_argument('--seed',type=int,default=2026);p.add_argument('--batch-size',type=int,default=512);args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    a=list(sample_episodes(args.episodes,args.seed));rng=random.Random(args.seed);rng.shuffle(a)
    # Split by episode (not by step) to limit correlated leakage.
    cut=int(args.episodes*.8);train=[r for r in a if r['episode']<cut];test=[r for r in a if r['episode']>=cut]
    # Additional challenging cases deliberately included only in evaluation.
    challenge=[]
    for seed_idx in range(220):
        s=fresh()
        for i,action in enumerate(['seed','rain','sun','dam','open']):
            nxt=transition(s,action)
            challenge.append(dict(episode=seed_idx,step=i,state=s.copy(),action=action,next_state=nxt.copy()))
            s=nxt
    def tensors(rows):
        return (torch.tensor([encode(r['state'],r['action']) for r in rows],dtype=torch.float32), torch.tensor([target(r['next_state']) for r in rows],dtype=torch.float32))
    tx,ty=tensors(train);vx,vy=tensors(test)
    model=Net();optim=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=args.epochs)
    history=[]
    for epoch in range(args.epochs):
        model.train();inds=torch.randperm(len(tx));tot=0
        for ix in inds.split(args.batch_size):
            logits=model(tx[ix]); cont=((logits[:,3]-ty[ix,3])**2+(logits[:,4]-ty[ix,4])**2).mean()
            idx=[0,1,2,5,6,7,8,9]
            classification=nn.functional.binary_cross_entropy_with_logits(logits[:,idx],ty[ix][:,idx]);loss=cont*7+classification
            optim.zero_grad();loss.backward();optim.step();tot+=loss.item()*len(ix)
        sch.step();model.eval()
        with torch.no_grad():
            z=model(vx); err=((z[:,3:5]-vy[:,3:5]).abs().mean()*100).item();accuracy=((z[:,[0,1,2,5,6,7,8,9]]>0)==(vy[:,[0,1,2,5,6,7,8,9]]>.5)).float().mean().item()
        history.append({'epoch':epoch+1,'loss':round(tot/len(tx),5),'continuous_mae_pct_points':round(err,3),'bool_accuracy':round(accuracy,5)})
        if (epoch+1)%10==0 or epoch==0:print(f'Epoch {epoch+1:>3} loss={history[-1]["loss"]:.4f} state MAE={err:.2f}pp bool={accuracy:.3%}',flush=True)
    model.eval()
    def infer(rows):
        x,_=tensors(rows)
        with torch.no_grad(): z=model(x);z[:,[0,1,2,5,6,7,8,9]]=torch.sigmoid(z[:,[0,1,2,5,6,7,8,9]])
        return [decode(row) for row in z.tolist()]
    def metrics(rows):
        pred=infer(rows)
        exact=sum(all((abs(p[k]-r['next_state'][k])<=2 if k in ('wet','eco') else p[k]==r['next_state'][k]) for k in FIELDS) for p,r in zip(pred,rows))/len(rows)
        boolacc=sum(p[k]==r['next_state'][k] for p,r in zip(pred,rows) for k in BOOL)/(len(rows)*len(BOOL))
        mae=sum(abs(p[k]-r['next_state'][k]) for p,r in zip(pred,rows) for k in ['wet','eco'])/(len(rows)*2)
        return {'samples':len(rows),'bool_accuracy':round(boolacc,4),'wet_eco_mae_points':round(mae,3),'all_fields_correct_within_2pp':round(exact,4)}
    val_pred=infer(test)
    baselines=[]
    for r in test:
        baselines.append(r['state']) # no-change baseline
    bacc=sum(p[k]==r['next_state'][k] for p,r in zip(baselines,test) for k in BOOL)/(len(test)*len(BOOL))
    bmae=sum(abs(p[k]-r['next_state'][k]) for p,r in zip(baselines,test) for k in ['wet','eco'])/(len(test)*2)
    output=ROOT/'artifacts';output.mkdir(exist_ok=True)
    weights=[]
    for layer in model.net:
        if isinstance(layer,nn.Linear):weights.append({'weights':layer.weight.detach().tolist(),'bias':layer.bias.detach().tolist()})
    data={'fields':FIELDS,'actions':ACTIONS,'layers':weights,'metrics':{'test':metrics(test),'challenge':metrics(challenge),'no_change_baseline':{'bool_accuracy':round(bacc,4),'wet_eco_mae_points':round(bmae,3)}},'history':history,'seed':args.seed,'train_samples':len(train),'test_samples':len(test),'challenge_definition':'seed → rain → sun → dam → open; 220 repeated scripted paths (not independent samples)'}
    (output/'model_weights.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')))
    torch.save({'state_dict':model.state_dict(),'fields':FIELDS,'actions':ACTIONS,'seed':args.seed},output/'world_dynamics.pt')
    (output/'test_samples.json').write_text(json.dumps([{**r,'prediction':p} for r,p in zip(test[:100],val_pred[:100])],ensure_ascii=False,indent=2))
    (output/'metrics.json').write_text(json.dumps(data['metrics'],indent=2))
    print('Metrics:',json.dumps(data['metrics'],indent=2))
if __name__=='__main__':main()
