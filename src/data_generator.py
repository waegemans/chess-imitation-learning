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
engine.configure(options={'Threads': 8})

board = chess.Board()


while True:
  if (board.is_game_over()):
      # reset board if game is over
      board.reset()
  fen_without_count = ' '.join(board.fen().split(' ')[:-2])

  move = None

  if fen_without_count not in d.keys():
        result = engine.play(board, chess.engine.Limit(depth=10))
        d[fen_without_count] = result.move.uci()
        w.writerow([fen_without_count, result.move.uci()])
        move = result.move
  if np.random.uniform() < 0.1:
      board.push(np.random.choice(list(board.legal_moves))[0])
  else:
      board.push_uci(d[fen_without_count])

engine.quit()

