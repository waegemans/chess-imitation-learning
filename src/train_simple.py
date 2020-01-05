import models
from dataset import ChessMoveDataset_pre_it_pov
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import progressbar
import data_util
import git
import os

import chess
import chess.engine


def init_weights(m):
  if type(m) == nn.Linear:
    torch.nn.init.xavier_normal_(m.weight)
    m.bias.data.fill_(0.01)

device = ('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
epochs = 20
batch_size = 1<<10
random_subset = None

githash = git.Repo(search_parent_directories=True).head.object.hexsha
log_dir = "output/" + githash + "/"

os.mkdir(log_dir)

log_file = open(log_dir+"out.csv", "w")

model = models.ssf_asf_res()
model.apply(init_weights)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.67, patience=0, verbose=True, threshold=1e-2)


trainset,valset = ChessMoveDataset_pre_it_pov(),ChessMoveDataset_pre_it_pov(mode='val')

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
val_iter = iter(val_loader)
log_file.write("epoch,batch_count,train_cross_entropy_loss,val_cross_entropy_loss,train_acc,val_acc,train_grads\n")

total_batch_count = 0

def sum_grads(model):
  train_params = filter(lambda p: p.requires_grad, model.parameters())
  return sum(ti.grad.detach().cpu().abs().sum().numpy() for ti in train_params)

def validate_batch():
  global val_iter
  x,y = None,None
  try:
    x,y = next(val_iter)
  except:
    val_iter = iter(val_loader)
    x,y = next(val_iter)
  
  x,y = x.to(device),y.to(device)
  model.eval()
  predicted = model(x)
  val_loss = nn.functional.cross_entropy(predicted, y,reduction='mean')
  val_acc = (predicted.detach().argmax(dim=1) == y).cpu().numpy().mean()

  return val_loss.detach().data.cpu().numpy(), val_acc


def train():
  global total_batch_count
  for x,y in progressbar.progressbar(train_loader,0,int(25930826*0.9/batch_size)+10):
    x,y = x.to(device),y.to(device)
    model.train()
    optimizer.zero_grad()
    #x,y = x.type(torch.float), y.type(torch.float)

    predicted = model(x)
    train_loss = nn.functional.cross_entropy(predicted, y,reduction='mean')
    train_loss.backward()
    train_grad = sum_grads(model)
    optimizer.step()

    train_acc = (predicted.detach().argmax(dim=1) == y).cpu().numpy().mean()

    val_loss = ''
    val_acc = ''

    if (total_batch_count % 10 == 0):
      val_loss,val_acc = validate_batch()

    log_file.write(','.join(map(str,[e,total_batch_count, train_loss.detach().data.cpu().numpy(), val_loss, train_acc, val_acc, train_grad]))+'\n')
    log_file.flush()

    total_batch_count += 1

def validate():
  samples = 0
  loss = 0
  for x,y in progressbar.progressbar(val_loader):
    x,y = x.to(device),y.to(device)
    model.eval()
    predicted = model(x).detach()
    loss += nn.functional.cross_entropy(predicted, y,reduction='sum')
    samples += len(x)
  return (loss/samples)

for e in range(epochs):
  torch.save(model, log_dir+'model_ep%d.nn'%e)
  print ("Epoch %d of %d:"%(e,epochs))

  train()
  val_loss = validate()
  print(val_loss)

  scheduler.step(val_loss)

torch.save(model, 'output/model_ep%d.nn'%epochs)


log_file.close()
