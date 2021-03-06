import chess
import numpy as np

def array_to_bitboard(x):
  return int(''.join(map(str,map(int,x[::-1]))),2)

def board_to_state(board):
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

def board_to_pieces(board):
  pieces = np.zeros((32,1+6+64), dtype=np.float)
  n = 0
  for color in (True,False):
    for piece in range(1,7):
      for square in board.pieces(piece,color):
        pieces[n,0] = color
        pieces[n,piece] = 1
        pieces[n,7+square] = 1
        n += 1
  return pieces

def state_to_cnn(state):
  res = np.zeros((17,8,8), np.float)
  res[:12,:,:] = state[:768].reshape((12,8,8))
  for i in range(5):
    res[12+i,:,:] += state[768+i]
  return res

def mirror_cnn(cnn):
  return cnn[:,:,::-1].copy()

def mirror_action(action):
  return action.reshape(8,8,8,8)[:,::-1,:,::-1].reshape(-1).copy()

def state_to_board(state):
  board = chess.Board()
  grid = np.array(state[:768].reshape(12,64))

  black = array_to_bitboard(grid[:6].sum(axis=0))
  white = array_to_bitboard(grid[6:].sum(axis=0))

  board.occupied_co = [black,white]
  board.occupied = black|white

  board.pawns = array_to_bitboard(grid[0]+grid[6])
  board.bishops = array_to_bitboard(grid[1]+grid[7])
  board.knights = array_to_bitboard(grid[2]+grid[8])
  board.rooks = array_to_bitboard(grid[3]+grid[9])
  board.queens = array_to_bitboard(grid[4]+grid[10])
  board.kings = array_to_bitboard(grid[5]+grid[11])

  castling_fen = ''

  if (state[770] == 1):
    castling_fen += 'K'
  if (state[771] == 1):
    castling_fen += 'Q'
  if (state[768] == 1):
    castling_fen += 'k'
  if (state[769] == 1):
    castling_fen += 'q'

  board.turn = (state[772]==1)

  if castling_fen == '':
    castling_fen = '-'

  board.set_castling_fen(castling_fen)

  return board


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
  return idx_to_uci(idx)

def idx_to_uci(idx):
  from_idx = idx//64
  to_idx = idx%64
  return chess.Move(from_idx, to_idx).uci()



def movelist_to_actionmask(movelist):
  res = np.zeros((64,64), np.float)
  for mv in movelist:
    res[mv.from_square, mv.to_square] = 1
  return res.reshape(-1)

def cpdict_to_loss_mask(cpdict):
  cp_loss = np.zeros((64,64), np.float)
  mask = np.zeros((64,64), np.float)
  max_cp = max(cpdict.values())
  min_cp = min(cpdict.values())
  cp_loss += min_cp - max_cp
  for uci,cp in cpdict.items():
    mv = chess.Move.from_uci(uci)
    mask[mv.from_square, mv.to_square] = 1
    cp_loss[mv.from_square, mv.to_square] = cp - max_cp
  return (cp_loss/100).reshape(-1), mask.reshape(-1)

def flip_uci(uci):
    uci_flipped = ""
    for c in uci:
        if c.isdigit():
            uci_flipped = uci_flipped + str(9-(ord(c)-ord('0')))
        else:
            uci_flipped = uci_flipped + c
    return uci_flipped
