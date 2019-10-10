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
		  nn.Linear(1024,4096),
                  nn.Softmax(dim=1)
		)

  def forward(self, state, action_mask):
    action_all = self.net(state)
    action_legal = action_all*action_mask
    return nn.functional.normalize(action_legal, p=1)


