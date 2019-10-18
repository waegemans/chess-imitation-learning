import torch
import torch.nn as nn

import numpy as np

class ssf_asf_1024_1024(nn.Module):
  def __init__(self):
    super(ssf_asf_1024_1024,self).__init__()
    self.net = nn.Sequential(
		  nn.Linear(773,1024),
		  nn.ReLU(),
		  nn.Linear(1024,1024),
		  nn.ReLU(),
		  nn.Linear(1024,4096)
		)

  def forward(self, state):
    return self.net(state)
    
    
class ssf_asf_512_512(nn.Module):
  def __init__(self):
    super(ssf_asf_512_512,self).__init__()
    self.net = nn.Sequential(
		  nn.Linear(773,512),
		  nn.ReLU(),
		  nn.Linear(512,512),
		  nn.ReLU(),
		  nn.Linear(512,4096)
		)

  def forward(self, state):
    return self.net(state)
    
def number_train_params(model):
  train_params = filter(lambda p: p.requires_grad, model.parameters())
  return sum(np.prod(ti.shape) for ti in train_params)


