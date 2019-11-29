import chess
import chess.engine
import numpy as np


engine = chess.engine.SimpleEngine.popen_uci("../../engine/stockfish_10_x64")

engine.configure(options={'Threads': 8})

board = chess.Board()

random_move_prob = 0.1

cnt = 0

while True:
  cnt += 1
  if (board.is_game_over()):
      # reset board if game is over
      board.reset()

  result = engine.play(board, limit=chess.engine.Limit(depth=20))
  move_uci = result.move.uci()

  print(board.fen())
  print(move_uci)

  if np.random.uniform() < random_move_prob:
      board.push(np.random.choice(list(board.legal_moves),1)[0])
  else:
      board.push(result.move)

engine.quit()