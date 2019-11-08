from models import ssf_asf_1024_1024, ssf_asf_512_512
from dataset import ChessMoveDataset
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
batch_size = 1<<12
random_subset = None

log_file = open("output/out.csv", "w")
move_file = open("output/moves.csv", "w")

#model = ssf_asf_1024_1024()
model = ssf_asf_512_512()
model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=0, verbose=True, threshold=1e-2)

base_dataset = ChessMoveDataset()

if random_subset is not None:
  base_dataset,_ = torch.utils.data.random_split(base_dataset, [random_subset,len(base_dataset)-(random_subset)])

n_train = int(0.9*len(base_dataset))
n_val = len(base_dataset)- n_train
dataset,valset = torch.utils.data.random_split(base_dataset, [n_train,n_val])
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)

log_file.write("epoch,batch_count,train_cross_entropy_loss,val_cross_entropy_loss,train_acc,val_acc\n")
move_file.write("epoch,fen,stockfish_uci,predicted_uci\n")

total_batch_count = 0

def validate_batch():
  x,y = next(iter(val_loader))
  model.eval()
  predicted = model(x)
  val_loss = nn.functional.cross_entropy(predicted, y.argmax(dim=1),reduce='mean')
  val_acc = (predicted.detach().argmax(dim=1) == y.argmax(dim=1)).numpy().mean()

  return val_loss.detach().data.numpy(), val_acc


def train():
  global total_batch_count
  for x,y in progressbar.progressbar(train_loader,0,len(train_loader)):
    model.train()
    optimizer.zero_grad()
    #x,y = x.type(torch.float), y.type(torch.float)

    predicted = model(x)
    train_loss = nn.functional.cross_entropy(predicted, y.argmax(dim=1),reduce='mean')
    train_loss.backward()
    optimizer.step()

    train_acc = (predicted.detach().argmax(dim=1) == y.argmax(dim=1)).numpy().mean()

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
  for x,y in progressbar.progressbar(val_loader,0,len(val_loader)):
    model.eval()
    predicted = model(x).detach()
    loss += nn.functional.cross_entropy(predicted, y.argmax(dim=1),reduce='sum')
    if samples < 1000:
      for xi,yi,pi in zip(x,y,predicted):
        board = data_util.state_to_board(xi)
        legal_mask = data_util.movelist_to_actionmask(board.legal_moves)
        agent_move = data_util.action_to_uci(yi)
        pred_move = data_util.action_to_uci(pi.numpy()*legal_mask)
        move_file.write(','.join(map(str,[e,board.fen(),agent_move,pred_move]))+'\n')
        move_file.flush()
    samples += len(x)
  return (loss/samples)

for e in range(epochs):
  print ("Epoch %d of %d:"%(e,epochs))

  train()
  val_loss = validate()
  print(val_loss)

  scheduler.step(val_loss)

  torch.save(model, 'output/model_ep%d.nn'%e)


log_file.close()
