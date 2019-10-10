import chess
import numpy as np

def board_to_state(board):
  black,white = board.occupied_co
  res = np.zeros(773, np.float)
  for j,color in enumerate(board.occupied_co):
    for i,pieces in enumerate([board.pawns, board.bishops, board.knights, board.rooks, board.queens, board.kings]):
      res[(i+j*6)*64:(i+j*6+1)*64] = np.array([int(c) for c in format(color&pieces, '064b')[::-1]])
  res[768] = board.has_kingside_castling_rights(False)
  res[769] = board.has_queenside_castling_rights(False)
  res[770] = board.has_kingside_castling_rights(True)
  res[771] = board.has_queenside_castling_rights(True)
  res[772] = board.turn
  return res

def move_to_action(move):
  res = np.zeros((64,64), np.float)
  res[move.from_square, move.to_square] = 1
  return res.reshape(-1)

def wuci_to_action(wuci):
  res = np.zeros((64,64), np.float)
  for uci,weight in wuci:
    move = chess.Move.from_uci(uci)
    res[move.from_square, move.to_square] = weight
  return res.reshape(-1)

def action_to_uci(action):
  idx = action.argmax()
  from_idx = idx//64
  to_idx = idx%64
  return chess.Move(from_idx, to_idx).uci()


def movelist_to_actionmask(movelist):
  res = np.zeros((64,64), np.float)
  for mv in movelist:
    res[mv.from_square, mv.to_square] = 1
  return res.reshape(-1) 
