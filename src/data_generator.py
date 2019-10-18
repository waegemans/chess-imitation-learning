import chess
import chess.engine
import csv
import numpy as np

d = {}
with open("data/moves.csv", "r") as csv_file:
    r = csv.reader(csv_file)
    for fen, move in r:
        d[fen] = move

w = csv.writer(open("data/moves.csv", "a"))

engine = chess.engine.SimpleEngine.popen_uci("../engine/stockfish_10_x64")

board = chess.Board()


while True:
  if (board.is_game_over()):
      # reset board if game is over
      board.reset()
  fen_without_count = ' '.join(board.fen().split(' ')[:-2])
  if fen_without_count in d.keys():
      # chose random move if position was already seen
      board.push(np.random.permutation(list(board.legal_moves))[0])
      continue
  result = engine.play(board, chess.engine.Limit(depth=10))
  d[fen_without_count] = result.move.uci()
  w.writerow([fen_without_count, result.move.uci()])
  board.push(result.move)

engine.quit()

