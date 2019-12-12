import numpy as np

import chess
import chess.engine
import chess.svg

import data_util

import models
import torch


def push_uci_queen(board,uci):
    try:
        board.push_uci(uci)
    except:
        #print("Defaulting to queen promotion")
        board.push_uci(uci+'q')

model = torch.load('output/model_ep7.nn')
model.eval()

board = chess.Board()
player_color = False
engine = chess.engine.SimpleEngine.popen_uci('../engine/stockfish_10_x64')

move = 0
while move < 900:
  if board.is_game_over():
      move += 1
      svg = chess.svg.board(board)
      with open('svg_0/move_'+str(move).zfill(3)+'.svg',"w") as f:
        f.write(svg)
      print(board.result())
      board.reset()
  move += 1
  svg = chess.svg.board(board)
  with open('svg_0/move_'+str(move).zfill(3)+'.svg',"w") as f:
      f.write(svg)
  color = board.turn
  if color == player_color:
      #mv = engine.play(board, limit=chess.engine.Limit(time=0.05))
      #board.push(mv.move)
      board.push(np.random.choice(list(board.legal_moves),1)[0])
      continue
      #print(list(map(lambda x: x.uci(), board.legal_moves)))
      #print("enter uci")
      #a = input()
      #board.push_uci(a)
      #continue

  state = data_util.board_to_state(board)
  y = model(torch.tensor(state,dtype=torch.float).unsqueeze(0))[0]
  y -= y.min()
  action = y * torch.tensor(data_util.movelist_to_actionmask(board.legal_moves),dtype=torch.float)
  #play according to prob
  '''
  np_action = action.detach().data.numpy()
  mv = np.random.choice(len(np_action),1,p=np_action/np_action.sum())
  mv_a = np.zeros_like(np_action)
  mv_a[mv] = 1
  #
  pred_uci = data_util.action_to_uci(mv_a)
  '''
  pred_uci = data_util.action_to_uci(action.detach().data)

  push_uci_queen(board,pred_uci)

move += 1
svg = chess.svg.board(board)
with open('svg_0/move_'+str(move).zfill(3)+'.svg',"w") as f:
    f.write(svg)

engine.quit()
