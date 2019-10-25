from models import ssf_asf_1024_1024, ssf_asf_512_512
from dataset import ChessMoveDataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import data_util
import sys
import progressbar

import chess
import chess.engine


def init_weights(m):
  if type(m) == nn.Linear:
    torch.nn.init.xavier_normal_(m.weight)
    m.bias.data.fill_(0.01)

epochs = 20
batch_size = 1<<12

csv_file = open("output/out.csv", "w")

model = ssf_asf_1024_1024()
#model = ssf_asf_512_512()
model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, 4, 0.3)

random_subset = None
base_dataset = ChessMoveDataset()

if random_subset is not None:
  base_dataset,_ = torch.utils.data.random_split(base_dataset, [random_subset,len(base_dataset)-(random_subset)])

n_train = int(0.8*len(base_dataset))
n_val = len(base_dataset)- n_train
dataset,valset = torch.utils.data.random_split(base_dataset, [n_train,n_val])
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)

csv_file.write("epoch,train_cross_entropy_loss,val_cross_entropy_loss,train_acc,val_acc\n")

total_batch_count = 0

for e in range(epochs):
  print ("Epoch %d of %d:"%(e,epochs))
  train_loss = 0
  val_loss = 0
  val_iter = iter(val_loader)
  for x,y in progressbar.progressbar(train_loader,0,len(train_loader)):
    model.train()
    optimizer.zero_grad()
    #x,y = x.type(torch.float), y.type(torch.float)

    predicted = model(x)
    loss = nn.functional.cross_entropy(predicted, y.argmax(dim=1),reduce='mean')
    loss.backward()
    optimizer.step()

    train_acc = (predicted.detach().argmax(dim=1) == y.argmax(dim=1)).numpy().mean()

    if total_batch_count%5==0:
      x_val,y_val = next(val_iter)
      model.eval()
      
      pred_val = model(x_val).detach()
      loss_val = nn.functional.cross_entropy(pred_val, y_val.argmax(dim=1),reduce='mean')

      val_acc = (pred_val.detach().argmax(dim=1) == y_val.argmax(dim=1)).numpy().mean()
      
      csv_file.write(','.join(map(str,[total_batch_count, loss.detach().data.numpy(), loss_val.detach().data.numpy(), train_acc, val_acc]))+'\n')
      csv_file.flush()

    total_batch_count += 1
  scheduler.step()
  torch.save(model, 'output/model_ep%d.nn'%e)


csv_file.close()
