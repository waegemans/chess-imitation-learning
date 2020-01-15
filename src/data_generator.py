import chess
import chess.engine
import numpy as np
import time
import os
import csv

n_threads = 8
depth = 18
gamma = 1
mate_score=10000

board = chess.Board()
engine = chess.engine.SimpleEngine.popen_uci("../engine/stockfish_10_x64_modern")
engine.configure(options={'Threads': n_threads})

game_id = 0

explored = {}
path = "data/depth%d_gamma%d/"%(depth,gamma)
os.makedirs(path,exist_ok=True)

with open(path+"moves.csv","r") as f:
    r = csv.reader(f)
    for fen,_ in r:
        explored[fen] = True

writer = csv.writer(open(path+"moves.csv","a"))

while True:
    fen_without_count = ' '.join(board.fen().split()[:-2])
    if board.is_game_over():
        board.reset()
        game_id += 1
    if not board.turn:
        board = board.mirror()

    if fen_without_count in explored.keys():
        board.push(np.random.choice(list(board.legal_moves)))
        continue
    
    info = engine.analyse(board,chess.engine.Limit(depth=depth),multipv=10,game=game_id)
    d = {}
    for i in info:
        d[i['pv'][0].uci()] = i['score'].pov(1).score(mate_score=mate_score)
    explored[fen_without_count] = True
    writer.writerow([fen_without_count,d])

engine.quit()