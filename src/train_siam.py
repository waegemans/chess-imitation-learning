import models
from dataset.statevalue import ChessMoveDataset_statevalue_it
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import progressbar
import git
import os
import util

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

model = models.cnn_siam().to(device)
#model = torch.load("output/0ab90067a02d8eb69c5aa4756eeed062d4872c5a/model_ep7.nn",map_location=device)

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, momentum=.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.67, patience=0, verbose=True, threshold=1e-2)

trainset,valset = ChessMoveDataset_statevalue_it(),ChessMoveDataset_statevalue_it(mode='val')

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
val_iter = iter(val_loader)
log_file.write("epoch,batch_count,train_cross_entropy_loss,val_cross_entropy_loss,train_acc,val_acc,train_grads\n")

def multi_cross_entropy(predicted, target, mask, topn=5):
  loss = 0
  midx = np.argpartition(-(target+mask).cpu().numpy(),topn)[:,:topn]
  w = torch.nn.functional.softmax(torch.tensor(np.take_along_axis(target.cpu().numpy(),midx,axis=1),device=device), dim=1)
  for i in range(topn):
    loss += (w[:,i]* nn.functional.cross_entropy(predicted, torch.tensor(midx[:,i],device=device),reduction='none')).mean()
  return loss



def loss_fcn(predicted, target):
  return nn.functional.cross_entropy(predicted,(10+target-util.shift(target)).float())

def acc_fnc(predicted,target):
    return ((predicted.argmax(dim=1)) == ((10+target-util.shift(target)))).cpu().numpy().mean()

total_batch_count = 0
running_train_loss = None

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
  val_loss = loss_fcn(predicted,y)
  val_acc = acc_fnc(predicted,y)

  return val_loss.detach().data.cpu().numpy(),val_acc


def train():
  global total_batch_count
  global running_train_loss
  for x,y in progressbar.progressbar(train_loader,max_value=len(trainset)//batch_size):
    x,y = x.to(device),y.to(device)
    perm = torch.randperm(x.size(0))
    x = x[perm]
    y = y[perm]
    model.train()
    optimizer.zero_grad()
    #x,y = x.type(torch.float), y.type(torch.float)

    predicted = model(x)
    train_loss = loss_fcn(predicted,y)
    train_acc = acc_fnc(predicted,y)
    train_loss.backward()
    train_grad = sum_grads(model)
    optimizer.step()

    val_loss = ''
    val_acc = ''

    if (total_batch_count % 10 == 0):
      val_loss,val_acc = validate_batch()

    log_file.write(','.join(map(str,[e,total_batch_count, train_loss.detach().data.cpu().numpy(), val_loss,train_acc,val_acc, train_grad]))+'\n')
    log_file.flush()

    total_batch_count += 1
    if running_train_loss is None:
      running_train_loss = train_loss.detach().data.cpu().numpy()
    running_train_loss = running_train_loss*0.9 + train_loss.detach().data.cpu().numpy()*0.1
    

def validate():
  pass

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
