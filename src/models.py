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

class RecNN(nn.Module):
  def __init__(self):
    super(RecNN,self).__init__()
    self.head = nn.Sequential(
      nn.Conv2d(17,196,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.Conv2d(196,196,kernel_size=3,padding=1),
      nn.ReLU(),
    )
    self.back = nn.Sequential(
      nn.Conv2d(196,196,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.Conv2d(196,196,kernel_size=3,padding=1)
    )
    self.tail = nn.Sequential(
      nn.Conv2d(196,196,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.Conv2d(196,64,kernel_size=1)
    )
    self.no_passes = 2

  def forward(self,x):
    h = self.head(x)
    out = h
    for i in range(self.no_passes):
      out = nn.functional.relu(self.back(out) + h)
    out = self.tail(out)
    return out.reshape((out.shape[0],-1))

class MultiAttentionHead(nn.Module):
  def __init__(self,query_dim,value_dim,q_in_dim,kv_in_dim,n_groups):
    super(MultiAttentionHead,self).__init__()
    self.query_dim = query_dim
    self.value_dim = value_dim
    self.n_groups = n_groups
    self.inv_sqrt_dim = query_dim ** -.5
    out_dim = q_in_dim

    self.query = nn.Linear(q_in_dim,self.query_dim*self.n_groups)
    self.key = nn.Linear(kv_in_dim,self.query_dim*self.n_groups)
    self.value = nn.Linear(kv_in_dim,self.value_dim*self.n_groups)
    self.project = nn.Linear(value_dim*n_groups,out_dim)
    self.ln = nn.LayerNorm(out_dim)

  def forward(self,Q,KV):
    # Q,KV: [batch,seq,in_dim]
    query = self.query(Q)
    key = self.key(KV)
    value = self.value(KV)
    batch_size = query.size(0)
    seq_size = query.size(1)
    seq_size_k = key.size(1)
    #query: [batch,seq,dim*groups]
    query = query.view(batch_size,seq_size,self.n_groups,self.query_dim).permute(0,2,1,3).reshape(-1,seq_size,self.query_dim)
    key = key.view(batch_size,seq_size_k,self.n_groups,self.query_dim).permute(0,2,3,1).reshape(-1,self.query_dim,seq_size_k)

    attention = torch.bmm(query,key)
    scores = nn.functional.softmax(attention*self.inv_sqrt_dim, dim=2)
    #attention: [batch*groups,seq,seq]
    value = value.view(batch_size,seq_size_k,self.value_dim,self.n_groups).permute(0,3,1,2).reshape(-1,seq_size_k,self.value_dim)
    attention = torch.bmm(scores,value)
    #attention: [batch*groups,seq,value_dim]
    attention = attention.view(batch_size,self.n_groups,seq_size,self.value_dim).permute(0,2,3,1).reshape(batch_size,seq_size,-1)

    attention = self.project(attention)
    attention += Q
    attention = self.ln(attention)

    return attention, scores

class ffn(nn.Module):
  def __init__(self,dim,hidden):
    super(ffn,self).__init__()
    self.model = nn.Sequential(
      nn.Linear(dim,hidden),
      nn.ReLU(),
      nn.Linear(hidden,dim)
    )
    self.ln = nn.LayerNorm(dim)
  
  def forward(self,X):
    out = self.model(X)
    out += X
    return self.ln(out)

class att_encoder(nn.Module):
    def __init__(self):
        super(att_encoder,self).__init__()
        self.e = nn.Linear(71,64)
        self.m1 = MultiAttentionHead(64,64,64,64,8)
        self.f1 = ffn(64,64)
        self.m2 = MultiAttentionHead(64,64,64,64,8)
        self.f2 = ffn(64,64)

    def forward(self, x):
        out = nn.functional.relu(self.e(x))
        out,_ = self.m1(out,out)
        out = self.f1(out)
        out,_ = self.m2(out,out)
        out = self.f1(out)
        return out

class att_decoder(nn.Module):
    def __init__(self):
        super(att_decoder,self).__init__()
        self.e = nn.Linear(64*2,64)
        self.m1 = MultiAttentionHead(64,64,64,64,8)
        self.f1 = ffn(64,64)
        self.m2 = MultiAttentionHead(64,64,64,64,8)
        self.f2 = ffn(64,64)
        
    def forward(self, x, enc):
        out = nn.functional.relu(self.e(x))
        out,_ = self.m1(out,enc)
        out = self.f1(out)
        out,_ = self.m2(out,enc)
        out = self.f1(out)
        return out
        
class Att_small(nn.Module):
  def __init__(self):
    super(Att_small,self).__init__()
    self.enc = att_encoder()
    self.dec = att_decoder()
    self.l = nn.Linear(64,1)

  def forward(self,pieces,moves):
      enc = self.enc(pieces)
      out = self.dec(moves,enc)
      out = self.l(out)
      return out.squeeze(-1)
    




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


