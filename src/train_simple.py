from models import ssf_asf_1024_1024
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

epochs = 100

csv_file = open("out.csv", "w")

model = ssf_asf_1024_1024()
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

random_subset = None
base_dataset = ChessMoveDataset()

if random_subset is not None:
  base_dataset,_ = torch.utils.data.random_split(base_dataset, [random_subset,len(base_dataset)-(random_subset)])

n_train = int(0.8*len(base_dataset))
n_val = len(base_dataset)- n_train
dataset,valset = torch.utils.data.random_split(base_dataset, [n_train,n_val])
train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(valset, batch_size=64, num_workers=8)

csv_file.write("epoch,train_cross_entropy_loss,val_cross_entropy_loss,train_acc,val_acc\n")

for e in range(epochs):
  train_loss = 0
  val_loss = 0
  train_correct = 0
  val_correct = 0
  for b,(x,y) in progressbar.progressbar(enumerate(train_loader),0,len(train_loader)):
    model.train()
    optimizer.zero_grad()
    x,y = x.type(torch.float), y.type(torch.float)

    predicted = model(x)
    loss = nn.functional.cross_entropy(predicted, y.argmax(dim=1),reduce='sum')
    loss.backward()
    optimizer.step()

    train_correct += (predicted.detach().argmax(dim=1) == y.argmax(dim=1)).sum().numpy()
    train_loss += loss.data.detach().numpy()

  for x,y in val_loader:
    model.eval()
    x,y = x.type(torch.float), y.type(torch.float)

    predicted = model(x)
    loss = nn.functional.cross_entropy(predicted, y.argmax(dim=1),reduce='sum')

    val_correct += (predicted.detach().argmax(dim=1) == y.argmax(dim=1)).sum().numpy()
    val_loss += loss.data.detach().numpy()
  csv_file.write(','.join(map(str,[e, train_loss/n_train, val_loss/n_val, train_correct/n_train, val_correct/n_val]))+'\n')
  print(','.join(map(str,[e, train_loss/n_train, val_loss/n_val, train_correct/n_train, val_correct/n_val])))

torch.save(model, 'model.nn')

csv_file.close()