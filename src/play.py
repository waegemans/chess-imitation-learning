import chess
import chess.engine

import numpy as np

n_games = 10
wins = 0
losses = 0
draws = 0

engine = chess.engine.SimpleEngine.popen_uci("./uci_wrapper.sh")
rengine = chess.engine.SimpleEngine.popen_uci("./uci_wrapper_random.sh")
board = chess.Board()

for i in range(n_games):
    board.reset()
    while not board.is_game_over():
        mv = None
        if board.turn:
            mv = engine.play(board,limit=chess.engine.Limit()).move

        else:
            mv = rengine.play(board,limit=chess.engine.Limit()).move
        board.push(mv)
    res = board.result()
    if res == '1-0':
        wins += 1
    elif res == '0-1':
        losses += 1
    else:
        draws += 1

print(wins,draws,losses)

