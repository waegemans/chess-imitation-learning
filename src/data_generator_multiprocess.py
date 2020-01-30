import chess
import chess.engine
import numpy as np
import queue
from multiprocessing import Queue, Process
import time
import zlib
import torch
import torch.nn as nn
import ast
import csv
import os


# number of threads
n_threads = 8
# search depth
depth = 18
# chance of playing a random move
gamma = 0.2
# cp score for a mate
mate_score=10000
# 
multipv=10

def analyze_game(fenqueue,dictqueue):
    engine = chess.engine.SimpleEngine.popen_uci("../engine/stockfish_10_x64_modern")
    while True:
        fen = fenqueue.get()
        board = chess.Board(fen)
        info = engine.analyse(board,chess.engine.Limit(depth=depth),multipv=multipv)
        d = {}
        for i in info:
            d[i['pv'][0].uci()] = i['score'].pov(1).score(mate_score=mate_score)
        dictqueue.put((fen,d))
    engine.quit()


if __name__ == "__main__":
    fenqueue = Queue(200)
    dictqueue = Queue(200)
    p = []

    movedict = {}

    path = "data/depth%d_gamma%f/"%(depth,gamma)
    os.makedirs(path,exist_ok=True)

    with open(path+"moves.csv","r") as f:
        r = csv.reader(f)
        for fen,dictstr in r:
            movedict[fen] = ast.literal_eval(dictstr)

    writer = csv.writer(open(path+"moves.csv","a"))

    for _ in range(n_threads):
        p.append(Process(target=analyze_game, args=(fenqueue,dictqueue,)))

    for pi in p:
        pi.start()

    start = time.time()
    
    board = chess.Board()
    while True:
        if board.is_game_over():
            board.reset()

        while not dictqueue.empty():
            fen,mvs = dictqueue.get()
            fen_without_count = ' '.join(fen.split()[:-2])
            if fen_without_count not in movedict.keys():
                movedict[fen_without_count] = mvs
                writer.writerow([fen_without_count,mvs])
        

        fen_without_count = ' '.join(board.fen().split()[:-2])
        if fen_without_count in movedict.keys():
            if np.random.rand() < gamma:
                board.push(np.random.choice(list(board.legal_moves)))
            else:
                d = movedict[fen_without_count]
                uci_list = []
                cp_list = []
                for uci,cp in d.items():
                    uci_list.append(uci)
                    cp_list.append(cp)
                
                uci = np.random.choice(uci_list,p=nn.functional.softmax(torch.tensor(cp_list,dtype=torch.float)).numpy())
                board.push_uci(uci)
        else:
            try:
                fenqueue.put(board.fen(),False,0.01)
            except queue.Full:
                pass
            board.push(np.random.choice(list(board.legal_moves)))
    
    for pi in p:
        pi.join()

    end = time.time()
    print(end-start)

