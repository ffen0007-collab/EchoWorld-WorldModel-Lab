"""Real small learned action-conditioned next-frame predictor. Synthetic frames from EchoWorld state semantics, not Genie."""
import json, pathlib, random, argparse
import numpy as np, torch
from torch import nn
from world import fresh, transition, ACTIONS
ROOT=pathlib.Path(__file__).resolve().parent.parent
SZ=32

def render(s):
 y,x=np.mgrid[:SZ,:SZ]; im=np.zeros((3,SZ,SZ),dtype=np.float32)
 # stylized world: fixed land, river, and objects; conditions change pixels
 im[:]=np.array([.29,.46,.35])[:,None,None]
 sky=y<10;im[:,sky]=np.array([.32,.52,.64])[:,None]
 river=np.abs(x-(14+4*np.sin(y/5)))< (2.5 if s['dam'] else 4.0)
 im[:,river]=np.array([.23,.51,.75])[:,None]
 if s['dam']:
  bar=(y>=17)&(y<20)&(x>=9)&(x<25);im[:,bar]=np.array([.55,.38,.22])[:,None]
 if s['seed']:
  patch=((x-25)**2+(y-23)**2)<(9 if s['grown'] else 3)
  im[:,patch]=np.array([.18,.8,.25] if s['grown'] else [.65,.53,.22])[:,None]
 tower=(x>=27)&(x<=30)&(y>=12)&(y<=21);im[:,tower]=np.array([.82,.71,.27] if s['lit'] else [.24,.26,.29])[:,None]
 if s['sun']:
  sun=(x-5)**2+(y-5)**2<9;im[:,sun]=np.array([1,.9,.42])[:,None]
 if s['rain']:
  drops=((x*3+y*7)%19==0)&(y<27);im[:,drops]=np.array([.7,.84,.95])[:,None]
 if s['wind']:
  wind=((y==7)|(y==9))&(x>17)&(x<26);im[:,wind]=np.array([.76,.88,.88])[:,None]
 # wet / eco visual encoding via patches of terrain, not visible text
 wetshade=.24*s['wet']/100;im[1,22:29,2:10]+=wetshade
 eco=.22*s['eco']/100;im[1,12:18,2:10]+=eco
 return np.clip(im,0,1)

class VisualNet(nn.Module):
 def __init__(self,hidden=32):
  super().__init__()
  self.net=nn.Sequential(nn.Conv2d(9,hidden,3,padding=1),nn.ReLU(),nn.Conv2d(hidden,hidden,3,padding=1),nn.ReLU(),nn.Conv2d(hidden,hidden,3,padding=1),nn.ReLU(),nn.Conv2d(hidden,3,1),nn.Sigmoid())
 def forward(self,frame,action):
  planes=action[:,:,None,None].expand(-1,-1,frame.shape[-2],frame.shape[-1]);return self.net(torch.cat([frame,planes],dim=1))

def build(n,seed):
 rng=random.Random(seed);xs=[];acts=[];ys=[]
 for i in range(n):
  s=fresh()
  for _ in range(rng.randint(0,24)):s=transition(s,rng.choice(ACTIONS))
  a=rng.randrange(6);nexts=transition(s,ACTIONS[a]);xs.append(render(s));ys.append(render(nexts));acts.append(a)
 return torch.from_numpy(np.stack(xs)),torch.nn.functional.one_hot(torch.tensor(acts),6).float(),torch.from_numpy(np.stack(ys))

def main():
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=12);p.add_argument('--train-size',type=int,default=2800);p.add_argument('--test-size',type=int,default=500);args=p.parse_args()
 torch.set_num_threads(2);torch.manual_seed(245);np.random.seed(245)
 x,a,y=build(args.train_size,503);vx,va,vy=build(args.test_size,3903)
 m=VisualNet();opt=torch.optim.AdamW(m.parameters(),lr=.002);history=[]
 for e in range(args.epochs):
  m.train();order=torch.randperm(len(x));running=0
  for idx in order.split(128):
   z=m(x[idx],a[idx]);loss=nn.functional.mse_loss(z,y[idx]);opt.zero_grad();loss.backward();opt.step();running+=loss.item()*len(idx)
  m.eval()
  with torch.no_grad():
   preds=torch.cat([m(xx,aa) for xx,aa in zip(vx.split(64),va.split(64))]);mse=((preds-vy)**2).mean().item();copy=((vx-vy)**2).mean().item();mae=(preds-vy).abs().mean().item()
  history.append({'epoch':e+1,'train_mse':round(running/len(x),6),'test_mse':round(mse,6),'copy_baseline_mse':round(copy,6)})
  print('visual epoch',e+1,'test MSE',round(mse,5),flush=True)
 torch.save({'state_dict':m.state_dict(),'resolution':SZ,'actions':ACTIONS},ROOT/'artifacts/visual_dynamics.pt')
 # comparison image: vertically stacked rows input/GT/pred, 6 cases
 from PIL import Image,ImageDraw
 idxs=[0,1,2,7,10,22,36,54]
 canvas=Image.new('RGB',(SZ*8*5,SZ*3*5+35),(17,28,44));d=ImageDraw.Draw(canvas)
 d.text((7,7),'INPUT',fill='white');d.text((7,37+SZ*5),'GROUND TRUTH',fill='white');d.text((7,37+SZ*10),'MODEL PREDICTION',fill='white')
 for j,i in enumerate(idxs):
  for k,img in enumerate([vx[i],vy[i],preds[i]]):
   im=Image.fromarray((img.permute(1,2,0).clamp(0,1).numpy()*255).astype('uint8')).resize((SZ*5,SZ*5),Image.Resampling.NEAREST)
   canvas.paste(im,(j*SZ*5,k*SZ*5+35))
 canvas.save(ROOT/'artifacts/visual_comparison.png')
 # save relevant sample as concise JS inputs for browser comparison
 examples=[]
 for i in idxs:
  examples.append({'action':ACTIONS[int(va[i].argmax())],'input':(vx[i].permute(1,2,0).numpy()*255).astype('uint8').reshape(-1).tolist(),'ground_truth':(vy[i].permute(1,2,0).numpy()*255).astype('uint8').reshape(-1).tolist(),'prediction':(preds[i].permute(1,2,0).detach().numpy()*255).astype('uint8').reshape(-1).tolist()})
 (ROOT/'artifacts/visual_examples.json').write_text(json.dumps(examples,separators=(',',':')))
 (ROOT/'artifacts/visual_metrics.json').write_text(json.dumps({'train_size':len(x),'test_size':len(vx),'test_mse':mse,'test_mae':mae,'copy_baseline_mse':copy,'history':history,'resolution':SZ,'method':'CNN input RGB previous frame + 6 action planes -> next RGB; synthetic deterministic simulator-generated images; independent test RNG seed'},indent=2))
if __name__=='__main__':main()
