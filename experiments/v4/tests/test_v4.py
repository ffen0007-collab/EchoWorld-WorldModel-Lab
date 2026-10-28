import unittest,sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
import torch
from world import fresh,transition,ACTIONS
from visual import render
from v4 import Model, make_episodes
class V4Tests(unittest.TestCase):
 def test_action_changes_simulator(self):
  self.assertNotEqual(transition(fresh(),'rain'), fresh())
 def test_frames_and_actions(self):
  f,a,s=make_episodes(2,77,steps=10)
  self.assertEqual(f[0].shape,(11,3,32,32));self.assertEqual(len(a[0]),10)
 def test_model_shapes_and_gradients(self):
  torch.manual_seed(1);m=Model();x=torch.tensor(render(fresh()))[None];a=torch.zeros(1,6);a[0,0]=1
  pred,h=m.next(x,a);self.assertEqual(tuple(pred.shape),(1,3,32,32));self.assertEqual(tuple(h.shape),(1,32,8,8));self.assertTrue(torch.isfinite(pred).all());pred.mean().backward();self.assertTrue(any(p.grad is not None for p in m.parameters()))
if __name__=='__main__':unittest.main()
