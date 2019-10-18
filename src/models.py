import torch
import torch.nn as nn

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


