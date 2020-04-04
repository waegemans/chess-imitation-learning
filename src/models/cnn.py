import torch
import torch.nn as nn
import util
from modules import cnn_res_block,inception_res_block

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

class unet_simple(nn.Module):
  def __init__(self):
    super(unet_simple,self).__init__()
    topc = 256
    midc = topc * 2
    lowc = midc * 2
    self.enc1 = nn.Sequential(
      nn.Conv2d(17,topc,kernel_size=3,padding=1),
      nn.BatchNorm2d(topc),
      nn.ReLU(),
      nn.Dropout2d(0.1),
      nn.Conv2d(topc,topc,kernel_size=3,padding=1),
      nn.BatchNorm2d(topc),
      nn.ReLU(),
      nn.Dropout2d(0.1),
    )
    self.enc2 = nn.Sequential(
      nn.Conv2d(topc,midc,kernel_size=3,padding=1),
      nn.BatchNorm2d(midc),
      nn.ReLU(),
      nn.Dropout2d(0.1),
      nn.Conv2d(midc,midc,kernel_size=3,padding=1),
      nn.BatchNorm2d(midc),
      nn.ReLU(),
      nn.Dropout2d(0.1),
    )
    self.enc3 = nn.Sequential(
      nn.Conv2d(midc,lowc,kernel_size=3,padding=1),
      nn.BatchNorm2d(lowc),
      nn.ReLU(),
      nn.Dropout2d(0.1),
      nn.Conv2d(lowc,lowc,kernel_size=3,padding=1),
      nn.BatchNorm2d(lowc),
      nn.ReLU(),
      nn.Dropout2d(0.1),
    )
    self.dec3 = nn.Sequential(
      nn.ConvTranspose2d(lowc,midc,kernel_size=2,stride=2),
      nn.BatchNorm2d(midc),
      nn.ReLU(),
      nn.Dropout2d(0.1),
    )
    self.dec2 = nn.Sequential(
      nn.Conv2d(lowc,midc,kernel_size=3,padding=1),
      nn.BatchNorm2d(midc),
      nn.ReLU(),
      nn.Dropout2d(0.1),
      nn.Conv2d(midc,midc,kernel_size=3,padding=1),
      nn.BatchNorm2d(midc),
      nn.ReLU(),
      nn.Dropout2d(0.1),
      nn.ConvTranspose2d(midc,topc,kernel_size=2,stride=2),
      nn.BatchNorm2d(topc),
      nn.ReLU(),
      nn.Dropout2d(0.1),
    )
    self.dec1 = nn.Sequential(
      nn.Conv2d(midc,topc,kernel_size=3,padding=1),
      nn.BatchNorm2d(topc),
      nn.ReLU(),
      nn.Dropout2d(0.1),
      nn.Conv2d(topc,topc,kernel_size=3,padding=1),
      nn.BatchNorm2d(topc),
      nn.ReLU(),
      nn.Dropout2d(0.1),
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

class flatten(nn.Module):
  def __init__(self):
    super(flatten,self).__init__()
  def forward(self,x):
    return x.reshape((x.shape[0],-1))


def cnn_res_small(hidden=283):
  return nn.Sequential(
    nn.Conv2d(17,hidden,kernel_size=3,padding=1),
    nn.BatchNorm2d(hidden),
    nn.ReLU(),

    cnn_res_block(hidden,kernel_size=3,dropout_p=0.1),
    cnn_res_block(hidden,kernel_size=3,dropout_p=0.1),

    
    nn.Conv2d(hidden,64,kernel_size=1),
    flatten()
  )

def inception_res_small(hidden=256):
  return nn.Sequential(
    nn.Conv2d(17,hidden,kernel_size=3,padding=1),
    nn.BatchNorm2d(hidden),
    nn.ReLU(),

    inception_res_block(hidden,dropout_p=0.1),
    inception_res_block(hidden,dropout_p=0.1),

    
    nn.Conv2d(hidden,64,kernel_size=1),
    flatten()
  )

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
  

class cnn_siam_bin(nn.Module):
  def __init__(self):
    super(cnn_siam_bin,self).__init__()
    self.fcn = fcn_small()
    self.fc1 = nn.Sequential(
        nn.Linear(64*64,128),
        nn.ReLU(),
        nn.Dropout(0.2)
        )
    self.fc2 = nn.Linear(256,1)

  def forward(self, x):
    out = self.fcn(x)
    out = self.fc1(out.reshape((out.shape[0],-1)))
    out = torch.cat((out,util.shift(out)),dim=1)
    return self.fc2(out).squeeze(-1)

