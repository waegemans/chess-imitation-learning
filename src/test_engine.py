import chess
import chess.engine
import numpy as np
import data_util



engine = chess.engine.SimpleEngine.popen_uci("../engine/stockfish_10_x64")

board = chess.Board()

d = {}
for mv in board.legal_moves:
  d[mv.uci()] = 0

for i in range(500):
  result = engine.play(board, chess.engine.Limit(time=0.1))
  d[result.move.uci()] += 1

d_filter = map(lambda e: (e[0],e[1]/500), filter(lambda item:item[1]!=0, d.items()))

print(list(d_filter))

engine.quit()
