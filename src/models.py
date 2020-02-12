import torch
import torch.nn as nn

import copy

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

class cnn_res_block(nn.Module):
    def __init__(self, hidden_channels, kernel_size):
        super(cnn_res_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(hidden_channels)
        )
    
    def forward(self, x):
        out = self.block(x)
        return nn.functional.relu(out+x)

class cnn_bare(nn.Module):
    def __init__(self):
        super(cnn_bare,self).__init__()
        self.model = nn.Sequential(
          nn.Conv2d(17,128,kernel_size=3,padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          
          #cnn_res_block(128,kernel_size=3),
          #cnn_res_block(128,kernel_size=3),
          
          nn.Conv2d(128,64,kernel_size=3,padding=1)
        )
    def forward(self, x):
      out = self.model(x)
      return out.reshape((out.shape[0],-1))

class cnn_simple(nn.Module):
    def __init__(self):
        super(cnn_simple,self).__init__()
        self.model = nn.Sequential(
          nn.Conv2d(17,512,kernel_size=5,padding=2),
          nn.BatchNorm2d(512),
          nn.ReLU(),
          
          cnn_res_block(512,kernel_size=5),
          cnn_res_block(512,kernel_size=5),
          
          nn.Conv2d(512,64,kernel_size=5,padding=2)
        )
    def forward(self, x):
      out = self.model(x)
      return out.reshape((out.shape[0],-1))

class add_dropout_cnn(nn.Module):
  def __init__(self,other):
    super(add_dropout_cnn,self).__init__()
    layers = []
    for x in other.children():
      layers.append(copy.deepcopy(x))
      if type(x) is cnn_res_block:
        layers.append(nn.Dropout2d(p=0.1))

    self.model = nn.Sequential(*layers)
  
  def forward(self, x):
    out = self.model(x)
    return out.reshape((out.shape[0],-1))

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

class unet_simple(nn.Module):
  def __init__(self):
    super(unet_simple,self).__init__()
    topc = 64
    midc = topc * 2
    lowc = midc * 2
    self.enc1 = nn.Sequential(
      nn.Conv2d(17,topc,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.Conv2d(topc,topc,kernel_size=3,padding=1),
      nn.ReLU()
    )
    self.enc2 = nn.Sequential(
      nn.Conv2d(topc,midc,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.Conv2d(midc,midc,kernel_size=3,padding=1),
      nn.ReLU()
    )
    self.enc3 = nn.Sequential(
      nn.Conv2d(midc,lowc,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.Conv2d(lowc,lowc,kernel_size=3,padding=1),
      nn.ReLU()
    )
    self.dec3 = nn.Sequential(
      nn.ConvTranspose2d(lowc,midc,kernel_size=2,stride=2),
      nn.ReLU()
    )
    self.dec2 = nn.Sequential(
      nn.Conv2d(lowc,midc,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.Conv2d(midc,midc,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(midc,topc,kernel_size=2,stride=2),
      nn.ReLU()
    )
    self.dec1 = nn.Sequential(
      nn.Conv2d(midc,topc,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.Conv2d(topc,topc,kernel_size=3,padding=1),
      nn.ReLU()
    )
    self.out = nn.Conv2d(topc,64,kernel_size=1)
  def forward(self,x):
    e1 = self.enc1(x)
    e2 = self.enc2(nn.functional.max_pool2d(e1, kernel_size=2,stride=2))
    e3 = self.enc3(nn.functional.max_pool2d(e2, kernel_size=2,stride=2))
    d3 = self.dec3(e3)
    d2 = self.dec2(torch.cat((d3,e2),dim=1))
    d1 = self.dec1(torch.cat((d2,e1),dim=1))
    out = self.out(d1)
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


