import models
from dataset import ChessMoveDataset_cp
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



device = ('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
epochs = 1000
batch_size = 1<<10
random_subset = None

githash = git.Repo(search_parent_directories=True).head.object.hexsha
log_dir = "output/" + githash + "/"

os.mkdir(log_dir)

log_file = open(log_dir+"out.csv", "w")

model = torch.load("output/0ab90067a02d8eb69c5aa4756eeed062d4872c5a/model_ep7.nn",map_location=device)

#freeze all but final layer
#for child in list(model.model.children())[:-1]:
#  for param in child.parameters():
#    param.requires_grad = False

#print(model)

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, momentum=.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.67, patience=0, verbose=True, threshold=1e-2)

ds = ChessMoveDataset_cp()

valn = int(len(ds)*0.1)//batch_size * batch_size

trainset,valset = torch.utils.data.random_split(ds,[len(ds)-valn,valn])
#trainset,valset = ChessMoveDataset_pre_it_pov_cnn(),ChessMoveDataset_pre_it_pov_cnn(mode='val')

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
val_iter = iter(val_loader)
log_file.write("epoch,batch_count,train_cross_entropy_loss,val_cross_entropy_loss,train_acc,val_acc,train_grads,train_min_cp,val_min_cp\n")

def multi_cross_entropy(predicted, target, mask, topn=5):
  loss = 0
  midx = np.argpartition(-target.cpu().numpy(),topn)[:,:topn]
  w = torch.nn.functional.softmax(torch.tensor(np.take_along_axis(target.cpu().numpy(),midx,axis=1),device=device), dim=1)
  for i in range(topn):
    loss += (w[:,i]* nn.functional.cross_entropy(predicted, torch.tensor(midx[:,i],device=device),reduction='none')).mean()
  return loss



def loss_fcn(predicted, target, mask):
  #mse = nn.functional.mse_loss(torch.flatten(predicted*mask),torch.flatten(target*mask),reduction='sum') / mask.sum()
  #hinge = (nn.functional.relu((predicted-target)*(1-mask))**2).sum() / (1-mask).sum()
  #cross_entropy = nn.functional.cross_entropy(predicted, target.argmax(dim=1),reduction='mean')
  probs = nn.functional.normalize(nn.functional.softmax(predicted, dim=1)*mask, p=1)
  if probs.sum() != len(probs):
    print(probs)
  avg_cp_loss = -(nn.functional.normalize(nn.functional.softmax(predicted, dim=1)*mask, p=1)*target).view(len(target),-1).sum(1).mean()
  return avg_cp_loss
  #return multi_cross_entropy(predicted, target, mask)

total_batch_count = 0
running_train_loss = None

def sum_grads(model):
  train_params = filter(lambda p: p.requires_grad, model.parameters())
  return sum(ti.grad.detach().cpu().abs().sum().numpy() for ti in train_params)

def validate_batch():
  global val_iter
  x,c,m = None,None,None
  try:
    x,c,m = next(val_iter)
  except:
    val_iter = iter(val_loader)
    x,c,m = next(val_iter)
  
  x,c,m = x.to(device),c.to(device),m.to(device)
  model.eval()
  predicted = model(x)
  val_loss = loss_fcn(predicted,c,m)
  val_acc = (predicted.detach().argmax(dim=1) == (c+m).argmax(dim=1)).cpu().numpy().mean()
  min_cp_loss = (c[torch.arange(len(predicted)),predicted.detach().argmax(dim=1)].mean().cpu().numpy())

  return val_loss.detach().data.cpu().numpy(), val_acc, min_cp_loss


def train():
  global total_batch_count
  global running_train_loss
  for x,c,m in progressbar.progressbar(train_loader):
    x,c,m = x.to(device),c.to(device),m.to(device)
    model.train()
    optimizer.zero_grad()
    #x,y = x.type(torch.float), y.type(torch.float)

    predicted = model(x)
    train_loss = loss_fcn(predicted,c,m)
    train_loss.backward()
    train_grad = sum_grads(model)
    optimizer.step()

    train_acc = (predicted.detach().argmax(dim=1) == (c+m).argmax(dim=1)).cpu().numpy().mean()
    min_cp_loss = (c[torch.arange(len(predicted)),predicted.detach().argmax(dim=1)].mean().cpu().numpy())
    val_loss = ''
    val_acc = ''
    val_min_cp_loss = ''

    if (total_batch_count % 10 == 0):
      val_loss,val_acc, val_min_cp_loss = validate_batch()

    log_file.write(','.join(map(str,[e,total_batch_count, train_loss.detach().data.cpu().numpy(), val_loss, train_acc, val_acc, train_grad, min_cp_loss, val_min_cp_loss]))+'\n')
    log_file.flush()

    total_batch_count += 1
    if running_train_loss is None:
      running_train_loss = train_loss.detach().data.cpu().numpy()
    running_train_loss = running_train_loss*0.9 + train_loss.detach().data.cpu().numpy()*0.1
    

def validate():
  samples = 0
  loss = 0
  for x,c,m in progressbar.progressbar(val_loader):
    x,c,m = x.to(device),c.to(device),m.to(device)
    model.eval()
    predicted = model(x).detach()
    loss += loss_fcn(predicted,c,m)
    samples += len(x)
  return (loss/samples)

for e in range(epochs):
  torch.save(model, log_dir+'model_ep%d.nn'%e)
  print ("Epoch %d of %d:"%(e,epochs))

  train()
  #val_loss = validate()
  #print(val_loss)
  print(running_train_loss)

  scheduler.step(running_train_loss)

torch.save(model, 'output/model_ep%d.nn'%epochs)


log_file.close()
