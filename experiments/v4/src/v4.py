"""EchoWorld v4: compact latent action-conditioned recurrent visual world model.
Public Genie 1/2 inspired, NOT Genie 4 replication.
"""
import argparse, json, math, random, pathlib, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw
from world import fresh,transition,ACTIONS
from visual import render
ROOT=pathlib.Path(__file__).resolve().parent.parent
ART=ROOT/'artifacts';ART.mkdir(exist_ok=True)

def make_episodes(n,seed,steps=54):
 r=random.Random(seed); frames=[];actions=[];states=[]
 for ep in range(n):
  s=fresh(); fs=[render(s)]; aa=[];ss=[s]
  for t in range(steps):
   # intentional action-relevant encounters + no-ops
   a=r.choice(ACTIONS) if r.random()<.66 else ACTIONS[(t+ep)%6]
   s=transition(s,a);aa.append(ACTIONS.index(a));fs.append(render(s));ss.append(s)
  frames.append(np.stack(fs).astype('float32'));actions.append(np.array(aa));states.append(ss)
 return frames,actions,states

class Encoder(nn.Module):
 def __init__(self,c=12):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(3,24,4,2,1),nn.ReLU(),nn.Conv2d(24,c,4,2,1),nn.Tanh())
 def forward(self,x):return self.net(x)
class Decoder(nn.Module):
 def __init__(self,c=12):
  super().__init__();self.net=nn.Sequential(nn.ConvTranspose2d(c,24,4,2,1),nn.ReLU(),nn.ConvTranspose2d(24,3,4,2,1),nn.Sigmoid())
 def forward(self,x):return self.net(x)
class LatentDynamics(nn.Module):
 def __init__(self,c=12,hidden=32):
  super().__init__();self.inp=nn.Conv2d(c+6,hidden,3,padding=1)
  self.gates=nn.Conv2d(hidden*2,hidden*2,3,padding=1)
  self.cand=nn.Conv2d(hidden*2,hidden,3,padding=1)
  self.out=nn.Conv2d(hidden,c,3,padding=1)
  self.hidden=hidden
 def forward(self,z,a,h=None):
  b,c,hh,ww=z.shape;planes=a[:,:,None,None].expand(b,6,hh,ww)
  x=F.relu(self.inp(torch.cat([z,planes],1)))
  if h is None:h=torch.zeros_like(x)
  reset,update=self.gates(torch.cat([x,h],1)).sigmoid().chunk(2,dim=1)
  candidate=torch.tanh(self.cand(torch.cat([x,reset*h],1)))
  h=(1-update)*h+update*candidate
  delta=self.out(h)
  return (z+0.4*torch.tanh(delta)).clamp(-1,1),h
class Model(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=Encoder();self.decoder=Decoder();self.dynamics=LatentDynamics()
 def next(self,frame,a,h=None):
  z=self.encoder(frame);nz,h=self.dynamics(z,a,h);return self.decoder(nz),h

def batch_one_step(episodes,acts,n,seed):
 r=random.Random(seed);xx=[];aa=[];yy=[]
 for i in range(n):
  k=r.randrange(len(episodes));t=r.randrange(len(acts[k]));xx.append(episodes[k][t]);yy.append(episodes[k][t+1]);aa.append(acts[k][t])
 return (torch.tensor(np.stack(xx)),F.one_hot(torch.tensor(aa),6).float(),torch.tensor(np.stack(yy)))

def evaluate(model,episodes,acts,steps=(1,10,50),max_eps=12):
 model.eval();res={}
 with torch.no_grad():
  for horizon in steps:
   results=[];copy_results=[];changed=[]
   for frames,actionseq in zip(episodes[:max_eps],acts[:max_eps]):
    x=torch.tensor(frames[:1]);first=x.clone();h=None
    for t in range(horizon):
     action=F.one_hot(torch.tensor([int(actionseq[t])]),6).float()
     x,h=model.next(x,action,h)
    target=torch.tensor(frames[horizon:horizon+1]);before=torch.tensor(frames[horizon-1:horizon]);
    results.append(float(F.mse_loss(x,target)));copy_results.append(float(F.mse_loss(first,target)))
    change=(target-before).abs().amax(1,keepdim=True)>.025
    if change.any():changed.append(float(((x-target)**2*change).sum()/(change.sum()*3)))
   res[str(horizon)]={'closed_loop_mse':sum(results)/len(results),'frozen_first_frame_mse':sum(copy_results)/len(copy_results),'changed_region_mse':sum(changed)/len(changed) if changed else None,'episodes':len(results)}
 return res

def save_panel(model,episodes,acts):
 model.eval();frames=episodes[0];actions=acts[0];samples=[0,1,3,10,25,50];preds={0:torch.tensor(frames[0:1])};h=None;cur=preds[0]
 with torch.no_grad():
  for t in range(50):
   a=F.one_hot(torch.tensor([int(actions[t])]),6).float();cur,h=model.next(cur,a,h)
   if t+1 in samples:preds[t+1]=cur.detach().clone()
 scale=5;sz=32*scale;im=Image.new('RGB',(len(samples)*sz,sz*2+42),(22,29,40));d=ImageDraw.Draw(im)
 for i,t in enumerate(samples):
  for row,arr in enumerate([frames[t],preds[t][0].numpy()]):
   img=Image.fromarray((np.clip(np.transpose(arr,(1,2,0)),0,1)*255).astype('uint8')).resize((sz,sz),Image.Resampling.NEAREST)
   im.paste(img,(i*sz,row*sz+36))
  d.text((i*sz+6,5),f't={t}',fill='white')
 im.save(ART/'rollout_comparison.png')

def main():
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=5);p.add_argument('--episodes',type=int,default=36);p.add_argument('--samples',type=int,default=1400);p.add_argument('--seed',type=int,default=2026);args=p.parse_args()
 torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed)
 tr,ta,_=make_episodes(args.episodes,args.seed);te,tea,_=make_episodes(12,args.seed+999)
 model=Model();opt=torch.optim.AdamW(model.parameters(),lr=.002);history=[];start=time.time()
 x,a,y=batch_one_step(tr,ta,args.samples,args.seed+1)
 for epoch in range(args.epochs):
  model.train();order=torch.randperm(len(x));rec=0
  for ids in order.split(64):
   bx,ba,by=x[ids],a[ids],y[ids]
   z=model.encoder(bx);reconstructed=model.decoder(z)
   next_z,h=model.dynamics(z,ba);pred=model.decoder(next_z)
   diff=(by-bx).abs().amax(1,keepdim=True);w=1+5*(diff>.025).float()
   loss=.5*F.mse_loss(reconstructed,bx)+((pred-by).square()*w).mean()+.02*(next_z-z).square().mean()
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();rec+=float(loss)*len(ids)
  ev=evaluate(model,te,tea,(1,),max_eps=8)
  history.append({'epoch':epoch+1,'train_loss':rec/len(x),'test_rollout_1_mse':ev['1']['closed_loop_mse']})
  print('epoch',epoch+1,history[-1],flush=True)
 report={'version':'v4.0.0','seed':args.seed,'train_episodes':args.episodes,'train_pairs':args.samples,'test_episodes':len(te),'epochs':args.epochs,'architecture':'32x32 RGB -> 8x8x12 latent CNN -> action-conditioned convolutional GRU residual -> RGB decoder','rollout':evaluate(model,te,tea,(1,10,50)), 'history':history,'runtime_seconds':round(time.time()-start,2),'scope':'Synthetic EchoWorld frames, no Genie 4 weights or API, exploratory CPU experiment. 50-step rollout seeded from one observed frame.'}
 torch.save({'model':model.state_dict(),'args':vars(args)},ART/'v4_latent_world_model.pt')
 (ART/'metrics.json').write_text(json.dumps(report,indent=2));save_panel(model,te,tea)
 print('SUMMARY',json.dumps(report['rollout']),flush=True)
if __name__=='__main__':main()
