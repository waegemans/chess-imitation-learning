from models import ssf_asf_2048_2048
from dataset import ChessMoveDataset_pre_it_pov
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import progressbar
import data_util

import chess
import chess.engine


def init_weights(m):
  if type(m) == nn.Linear:
    torch.nn.init.xavier_normal_(m.weight)
    m.bias.data.fill_(0.01)

epochs = 20
batch_size = 1<<10
random_subset = None

log_file = open("output/out.csv", "w")

model = ssf_asf_2048_2048()
#model = ssf_asf_512_512_512()
model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=0, verbose=True, threshold=1e-2)


trainset,valset = ChessMoveDataset_pre_it_pov(),ChessMoveDataset_pre_it_pov(mode='val')

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
val_iter = iter(val_loader)
log_file.write("epoch,batch_count,train_cross_entropy_loss,val_cross_entropy_loss,train_acc,val_acc\n")

total_batch_count = 0

def validate_batch():
  global val_iter
  x,y = None,None
  try:
    x,y = next(val_iter)
  except:
    val_iter = iter(val_loader)
    x,y = next(val_iter)
  model.eval()
  predicted = model(x)
  val_loss = nn.functional.cross_entropy(predicted, y,reduction='mean')
  val_acc = (predicted.detach().argmax(dim=1) == y).numpy().mean()

  return val_loss.detach().data.numpy(), val_acc


def train():
  global total_batch_count
  for x,y in progressbar.progressbar(train_loader,0,int(25930826*0.9/batch_size)+10):
    model.train()
    optimizer.zero_grad()
    #x,y = x.type(torch.float), y.type(torch.float)

    predicted = model(x)
    train_loss = nn.functional.cross_entropy(predicted, y,reduction='mean')
    train_loss.backward()
    optimizer.step()

    train_acc = (predicted.detach().argmax(dim=1) == y).numpy().mean()

    val_loss = ''
    val_acc = ''

    if (total_batch_count % 10 == 0):
      val_loss,val_acc = validate_batch()

    log_file.write(','.join(map(str,[e,total_batch_count, train_loss.detach().data.numpy(), val_loss, train_acc, val_acc]))+'\n')
    log_file.flush()

    total_batch_count += 1

def validate():
  samples = 0
  loss = 0
  for x,y in progressbar.progressbar(val_loader):
    model.eval()
    predicted = model(x).detach()
    loss += nn.functional.cross_entropy(predicted, y,reduction='sum')
    samples += len(x)
  return (loss/samples)

for e in range(epochs):
  torch.save(model, 'output/model_ep%d.nn'%e)
  print ("Epoch %d of %d:"%(e,epochs))

  train()
  val_loss = validate()
  print(val_loss)

  scheduler.step(val_loss)

torch.save(model, 'output/model_ep%d.nn'%epochs)


log_file.close()
