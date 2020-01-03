import torch
import torch.nn as nn

import numpy as np

class fc_res_block(nn.Module):
    def __init__(self, hidden_size):
        super(fc_res_block,self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            )
        
    def forward(self, x):
        out = self.block(x)
        return nn.functional.relu(out+x)

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
 
class ssf_asf_1024_1024_1024(nn.Module):
  def __init__(self):
    super(ssf_asf_1024_1024_1024,self).__init__()
    self.net = nn.Sequential(
		  nn.Linear(773,1024),
		  nn.ReLU(),
		  nn.Linear(1024,1024),
		  nn.ReLU(),
		  nn.Linear(1024,1024),
		  nn.ReLU(),
		  nn.Linear(1024,4096)
		)

  def forward(self, state):
    return self.net(state)


class ssf_asf_2048_2048(nn.Module):
  def __init__(self):
    super(ssf_asf_2048_2048,self).__init__()
    self.net = nn.Sequential(
		  nn.Linear(773,2048),
		  nn.ReLU(),
		  nn.Linear(2048,2048),
		  nn.ReLU(),
		  nn.Linear(2048,4096)
		)

  def forward(self, state):
    return self.net(state)


class ssf_asf_2048_res(nn.Module):
  def __init__(self):
    super(ssf_asf_2048_res,self).__init__()
    self.net = nn.Sequential(
		  nn.Linear(773,4096),
		  nn.ReLU(),

      fc_res_block(4096),

      fc_res_block(4096),

      fc_res_block(4096),

      fc_res_block(4096),

		  nn.Linear(4096,4096)
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
     
class ssf_asf_512_512_512(nn.Module):
  def __init__(self):
    super(ssf_asf_512_512_512,self).__init__()
    self.net = nn.Sequential(
		  nn.Linear(773,512),
		  nn.ReLU(),
		  nn.Linear(512,512),
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


