from models import ssf_asf_1024_1024
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import data_util
import sys

import chess
import chess.engine

def init_weights(m):
  if type(m) == nn.Linear:
    torch.nn.init.xavier_normal_(m.weight)
    m.bias.data.fill_(0.01)

epochs = 1000

board = chess.Board()
engine = chess.engine.SimpleEngine.popen_uci("../engine/stockfish_10_x64")

model = ssf_asf_1024_1024()
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("sample_nr,mse_loss,mv_count,stockfish,predicted")
running_loss = 1

for e in range(epochs):
  if len(board.move_stack) == 2 or board.is_game_over():
    board.reset()
  move = engine.play(board, chess.engine.Limit(time=0.1)).move
  
  
  state = torch.tensor(data_util.board_to_state(board),dtype=torch.float32).unsqueeze_(0)
  action = torch.tensor(data_util.move_to_action(move),dtype=torch.float32).unsqueeze_(0)
  action_mask = torch.tensor(data_util.movelist_to_actionmask(board.legal_moves),dtype=torch.float32).unsqueeze_(0)
  

  model.train()
  optimizer.zero_grad()

  predicted = model(state,action_mask)
  loss = nn.functional.mse_loss(predicted[action_mask==1], action[action_mask==1], reduction='mean')
  loss.backward()
  optimizer.step()
  print(','.join(map(str,[e, loss.data.detach().numpy(), len(board.move_stack), move.uci(), data_util.action_to_uci(predicted.data.detach())])))
  running_loss = running_loss*0.9 + loss.data.detach().numpy()*0.1
  print(e, loss.data.detach().numpy(), running_loss, file=sys.stderr)
  board.push(move)
engine.quit()
  
  
