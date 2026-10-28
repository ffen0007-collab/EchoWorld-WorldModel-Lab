"""EchoWorld v3: train/ablate real residual and ConvGRU image predictors on simulator imagery."""
import os, json, random, time, argparse, pathlib, sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageDraw
from world import fresh,transition,ACTIONS
from visual import render,VisualNet
ROOT=pathlib.Path(__file__).resolve().parent.parent
ART=ROOT/'artifacts'; ART.mkdir(exist_ok=True)
SZ=32

def seeded(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)

def episode_dataset(n,seed,history=4,changed_focus=.7):
 """Episode-independent data, oversamples actual image changes, with some difficult conditional transitions."""
 rng=random.Random(seed); out=[]
 for ep in range(n):
  s=fresh(); frames=[render(s)]; actions=[]
  for t in range(rng.randint(15,31)):
   # targeted conditions create meaningful transitions more frequently
   if rng.random()<.55:
    choices=['rain','sun','seed','dam','open','wind']
    a=rng.choice(choices)
   else:a=rng.choice(ACTIONS)
   nxt=transition(s,a);img=render(nxt);diff=float(np.abs(img-frames[-1]).mean())
   seq=np.stack(([frames[0]]*max(0,history-len(frames)))+frames[-history:])
   out.append(dict(x=seq,y=img,a=ACTIONS.index(a),changed=diff>0.001,episode=ep,step=t, state=s.copy(),next=nxt.copy()))
   s=nxt;frames.append(img);actions.append(a)
 # stratified oversampling without changing test distributions
 if changed_focus>0:
  changed=[z for z in out if z['changed']];static=[z for z in out if not z['changed']]
  target=int(len(out)*changed_focus)
  if changed and len(changed)<target:
   out=out+rng.choices(changed,k=target-len(changed))
 rng.shuffle(out)
 return out

def tensors(data):
 x=torch.from_numpy(np.stack([z['x'] for z in data])).float()
 y=torch.from_numpy(np.stack([z['y'] for z in data])).float()
 a=F.one_hot(torch.tensor([z['a'] for z in data]),6).float()
 return x,a,y

class BaselineCNN(VisualNet):
 def forward(self,x,a):return super().forward(x[:,-1],a)

class ResidualCNN(nn.Module):
 def __init__(self,hidden=32):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(9,hidden,3,padding=1),nn.ReLU(),nn.Conv2d(hidden,hidden,3,padding=1),nn.ReLU(),nn.Conv2d(hidden,hidden,3,padding=1),nn.ReLU(),nn.Conv2d(hidden,3,3,padding=1))
 def forward(self,x,a):
  prev=x[:,-1];planes=a[:,:,None,None].expand(-1,-1,SZ,SZ)
  return torch.clamp(prev+0.5*torch.tanh(self.net(torch.cat([prev,planes],1))),0,1)

class ConvGRUCell(nn.Module):
 def __init__(self,in_ch,hidden):
  super().__init__();self.hidden=hidden;self.gates=nn.Conv2d(in_ch+hidden,hidden*2,3,padding=1);self.candidate=nn.Conv2d(in_ch+hidden,hidden,3,padding=1)
 def forward(self,x,h):
  if h is None:h=torch.zeros((x.shape[0],self.hidden,SZ,SZ),device=x.device)
  reset,update=self.gates(torch.cat([x,h],1)).sigmoid().chunk(2,dim=1)
  cand=self.candidate(torch.cat([x,reset*h],1)).tanh();return (1-update)*h+update*cand

class ResidualConvGRU(nn.Module):
 def __init__(self,hidden=24):
  super().__init__();self.enc=nn.Sequential(nn.Conv2d(3,hidden,3,padding=1),nn.ReLU());self.cell=ConvGRUCell(hidden,hidden)
  self.head=nn.Sequential(nn.Conv2d(hidden+6,hidden,3,padding=1),nn.ReLU(),nn.Conv2d(hidden,3,3,padding=1))
 def forward(self,x,a):
  h=None
  for t in range(x.shape[1]):h=self.cell(self.enc(x[:,t]),h)
  planes=a[:,:,None,None].expand(-1,-1,SZ,SZ)
  delta=self.head(torch.cat([h,planes],1));return torch.clamp(x[:,-1]+0.5*torch.tanh(delta),0,1)

# perceptual object masks on semantic image regions; pixel-based visibility proxy, not pretrained foundation vision assessment
MASKS={
 'tower': (slice(12,22),slice(27,31)),
 'dam': (slice(17,20),slice(9,25)),
 'vegetation': (slice(20,27),slice(21,30)),
 'river': (slice(0,32),slice(9,21)),
}
def score(pred,y,x,changed):
 err=(pred-y).square();mask=(y-x[:,-1]).abs().amax(1,keepdim=True)>.025
 change_err=(err*mask).sum().item()/max(1,(mask.sum().item()*3));static_err=(err*(~mask)).sum().item()/max(1,(~mask).sum().item()*3)
 result={'mse':err.mean().item(),'changed_mse':change_err,'static_mse':static_err,'changed_pixel_fraction':mask.float().mean().item()}
 # Region mean color proxy: if structure appears in real region, check visual matching via region pixel MSE
 for name,(ys,xs) in MASKS.items():result[f'{name}_region_mse']=err[:,:,ys,xs].mean().item()
 # thresholded meaningful-object appearance detection from visible pixels with exact representative colors
 # dam structure local brown; tower illuminated gold; vegetation green. Evaluate observational semantic consistency
 defs={'dam':((slice(17,20),slice(9,25)),lambda z:(z[:,0]>.43)&(z[:,1]>.25)&(z[:,1]<.49)&(z[:,2]<.32)),
       'tower_lit':((slice(12,22),slice(27,31)),lambda z:(z[:,0]>.65)&(z[:,1]>.52)&(z[:,2]<.42)),
       'grown_plant':((slice(20,27),slice(21,30)),lambda z:(z[:,1]>.66)&(z[:,0]<.38)&(z[:,2]<.45))}
 for name,((ys,xs),fn) in defs.items():
  truth=fn(y[:,:,ys,xs]).float().mean((1,2))>.15
  found=fn(pred[:,:,ys,xs]).float().mean((1,2))>.15
  result[name+'_presence_accuracy']=(truth==found).float().mean().item()
  result[name+'_positive_count']=int(truth.sum())
 return result

def evaluate(model,data,batch_size=128,keep=False):
 if model is not None:model.eval()
 preds=[]
 with torch.no_grad():
  for ix in range(0,len(data),batch_size):
   x,a,y=tensors(data[ix:ix+batch_size]);z=model(x,a) if model is not None else x[:,-1];preds.append(z)
 x,a,y=tensors(data);pred=torch.cat(preds)
 return score(pred,y,x,None),(pred,x,y) if keep else None

def train_model(model,train,val,epochs=6,weighted=False,lr=.002):
 opt=torch.optim.AdamW(model.parameters(),lr=lr);hist=[]
 x,a,y=tensors(train)
 for epoch in range(epochs):
  model.train();order=torch.randperm(len(x));total=0.
  for ids in order.split(96):
   out=model(x[ids],a[ids]);diff=(y[ids]-x[ids,-1]).abs().amax(1,keepdim=True)
   mask=(diff>.025).float(); weight=1+7*mask if weighted else torch.ones_like(mask)
   loss=((out-y[ids]).square()*weight).mean()
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();total+=loss.item()*len(ids)
  v,_=evaluate(model,val);hist.append({'epoch':epoch+1,'train_weighted_loss':total/len(train),'val_mse':v['mse'],'val_changed_mse':v['changed_mse']})
  print(type(model).__name__, 'weighted' if weighted else 'plain',epoch+1, 'val',round(v['mse'],5),'change',round(v['changed_mse'],5),flush=True)
 return hist

def plots(models,test):
 # select only examples with changed pixels, and diverse actions
 samples=[]
 for act in range(6):
  candidates=[z for z in test if z['a']==act and z['changed']]
  if candidates:samples.append(candidates[0])
 px=128; W=(3+len(models))*px;H=len(samples)*px+35
 out=Image.new('RGB',(W,H),(22,29,41));d=ImageDraw.Draw(out)
 names=['Input','Ground truth','Copy']+list(models)
 for j,name in enumerate(names):d.text((j*px+6,8),name,fill='white')
 examples=[]
 for k,entry in enumerate(samples):
  x,a,y=tensors([entry]);images=[x[0,-1],y[0],x[0,-1]]
  for name,model in models.items():
   with torch.no_grad():images.append(model(x,a)[0])
  for j,im in enumerate(images):
   pic=Image.fromarray((im.permute(1,2,0).detach().clamp(0,1).numpy()*255).astype('uint8')).resize((px,px),Image.Resampling.NEAREST)
   out.paste(pic,(j*px,35+k*px))
  d.text((3,37+k*px),ACTIONS[entry['a']],fill='white',stroke_width=1,stroke_fill='black')
  examples.append(dict(action=ACTIONS[entry['a']], images={name:(im.permute(1,2,0).detach().clamp(0,1).numpy()*255).astype('uint8').reshape(-1).tolist() for name,im in zip(names,images)}))
 out.save(ART/'ablation_comparison.png');(ART/'examples.json').write_text(json.dumps(examples))

def main():
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=6);p.add_argument('--train-episodes',type=int,default=180);p.add_argument('--val-episodes',type=int,default=30);p.add_argument('--test-episodes',type=int,default=35);args=p.parse_args()
 seeded(2026)
 train=episode_dataset(args.train_episodes,2026,changed_focus=.73)
 val=episode_dataset(args.val_episodes,8091,changed_focus=0)
 test=episode_dataset(args.test_episodes,9027,changed_focus=0)
 # No sample from test episodes in training, test distribution not oversampled
 basemetrics,_=evaluate(None,test)
 variants=[('cnn_baseline',lambda:BaselineCNN(),False),('residual',lambda:ResidualCNN(),False),('residual_weighted',lambda:ResidualCNN(),True),('convgru_weighted',lambda:ResidualConvGRU(),True)]
 models={};results={'config':vars(args),'data':{'train_samples':len(train),'val_samples':len(val),'test_samples':len(test),'train_changed_fraction':sum(z['changed'] for z in train)/len(train),'test_changed_fraction':sum(z['changed'] for z in test)/len(test),'test_seed':9027,'split':'independent generated episodes; test not oversampled'},'variants':{'copy_previous_frame':{'metrics':basemetrics,'parameters':0}}}
 for name,ctor,weighted in variants:
  model=ctor();t0=time.time();hist=train_model(model,train,val,args.epochs,weighted)
  metrics,_=evaluate(model,test)
  results['variants'][name]={'metrics':metrics,'parameters':sum(p.numel() for p in model.parameters()),'training_seconds':round(time.time()-t0,2),'history':hist}
  torch.save({'state_dict':model.state_dict(),'variant':name,'actions':ACTIONS,'frame_size':SZ,'history':4,'architecture':type(model).__name__},ART/(name+'.pt'))
  models[name]=model
  (ART/'ablation_results.json').write_text(json.dumps(results,indent=2))
 plots(models,test)
 print('FINISHED',json.dumps({k:v['metrics']['mse'] for k,v in results['variants'].items()}))
if __name__=='__main__':main()
