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

class cnn_simple(nn.Module):
    def __init__(self):
        super(cnn_simple,self).__init__()
        self.model = nn.Sequential(
          nn.Conv2d(17,512,kernel_size=5,padding=2),
          nn.ReLU(),
          nn.Conv2d(512,512,kernel_size=5,padding=2),
          nn.ReLU(),
          nn.Conv2d(512,512,kernel_size=5,padding=2),
          nn.ReLU(),
          nn.Conv2d(512,64,kernel_size=5,padding=2)
        )
    def forward(self, x):
      out = self.model(x)
      return out.reshape((out.shape[0],-1))


def small():
  return nn.Sequential(
    nn.Linear(773,512),
    nn.ReLU(),

    fc_res_block(512),
    fc_res_block(512),
    fc_res_block(512),
    fc_res_block(512),

    nn.Linear(512,4096)
	)
def ssf_asf_res():
  return nn.Sequential(
    nn.Linear(773,8192),
    nn.ReLU(),

    fc_res_block(8192),

    fc_res_block(8192),

    nn.Linear(8192,4096)
	)
  
def number_train_params(model):
  train_params = filter(lambda p: p.requires_grad, model.parameters())
  return sum(np.prod(ti.shape) for ti in train_params)


