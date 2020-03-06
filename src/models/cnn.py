import torch
import torch.nn as nn
import util
from modules import cnn_res_block

class cnn_alpha(nn.Module):
    def __init__(self):
        super(cnn_alpha,self).__init__()
        self.model = nn.Sequential(
          nn.Conv2d(17,256,kernel_size=3,padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          
          cnn_res_block(256,kernel_size=3),
          cnn_res_block(256,kernel_size=3),
          cnn_res_block(256,kernel_size=3),
          cnn_res_block(256,kernel_size=3),
          cnn_res_block(256,kernel_size=3),
          cnn_res_block(256,kernel_size=3),
          cnn_res_block(256,kernel_size=3),
          cnn_res_block(256,kernel_size=3),
          cnn_res_block(256,kernel_size=3),
          cnn_res_block(256,kernel_size=3),
          cnn_res_block(256,kernel_size=3),
          cnn_res_block(256,kernel_size=3),
          cnn_res_block(256,kernel_size=3),
          cnn_res_block(256,kernel_size=3),
          cnn_res_block(256,kernel_size=3),
          cnn_res_block(256,kernel_size=3),
          cnn_res_block(256,kernel_size=3),
          cnn_res_block(256,kernel_size=3),
          
          nn.Conv2d(256,256,kernel_size=3,padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.Conv2d(256,64,kernel_size=1)
        )
    def forward(self, x):
      out = self.model(x)
      return out.reshape((out.shape[0],-1))


class cnn_alpha_small(nn.Module):
    def __init__(self):
        super(cnn_alpha_small,self).__init__()
        self.model = nn.Sequential(
          nn.Conv2d(17,256,kernel_size=3,padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          
          cnn_res_block(256,kernel_size=3),
          
          nn.Conv2d(256,256,kernel_size=3,padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.Conv2d(256,64,kernel_size=1)
        )
    def forward(self, x):
      out = self.model(x)
      return out.reshape((out.shape[0],-1))

def fcn_small():
    return nn.Sequential(
          nn.Conv2d(17,256,kernel_size=3,padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          
          cnn_res_block(256,kernel_size=3),
          
          nn.Conv2d(256,256,kernel_size=3,padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.Conv2d(256,64,kernel_size=1)
        )

class cnn_value(nn.Module):
  def __init__(self):
    super(cnn_value,self).__init__()
    self.fcn = fcn_small()
    self.lin = nn.Linear(64*64,1)
    self.fc = nn.Sequential(
        nn.Linear(64*64,128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128,1)
        )
    

  def forward(self, x):
    out = self.fcn(x)
    return self.fc(out.reshape((out.shape[0],-1))).squeeze(-1)

class cnn_disc(nn.Module):
  def __init__(self):
    super(cnn_disc,self).__init__()
    self.fcn = fcn_small()
    self.fc = nn.Sequential(
        nn.Linear(64*64,128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128,11)
        )

  def forward(self, x):
    out = self.fcn(x)
    return self.fc(out.reshape((out.shape[0],-1)))

class cnn_siam(nn.Module):
  def __init__(self):
    super(cnn_siam,self).__init__()
    self.fcn = fcn_small()
    self.fc1 = nn.Sequential(
        nn.Linear(64*64,128),
        nn.ReLU(),
        nn.Dropout(0.2)
        )
    self.fc2 = nn.Linear(256,21)

  def forward(self, x):
    out = self.fcn(x)
    out = self.fc1(out.reshape((out.shape[0],-1)))
    out = torch.cat((out,util.shift(out)),dim=1)
    return self.fc2(out).squeeze(-1)

