import pandas as pd
import numpy as np

import chess
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci('../engine/stockfish_10_x64')

moves = pd.read_csv('output/tmpmoves.csv')
#shuffle moves
moves = moves.sample(frac=1).reset_index(drop=True)


log_file = open("output/centipawnloss_agg.csv", "w")
log_file.write('epoch,loss_agg\n')

def push_uci_queen(board,uci):
    try:
        board.push_uci(uci)
    except:
        print("Defaulting to queen promotion")
        board.push_uci(uci+'q')

for key,group in moves.groupby('epoch'):
    centipawnloss = 0
    count = 0
    for index, row in group.iterrows():
        b = chess.Board(fen=row['fen'])
        color = b.turn
        sf_an = engine.analyse(b,limit=chess.engine.Limit(depth=10))

        push_uci_queen(b,row['predicted_uci'])
        pr_an = engine.analyse(b,limit=chess.engine.Limit(depth=10))

        sf_score = sf_an['score'].pov(color).score(mate_score=1000000)
        pr_score = pr_an['score'].pov(color).score(mate_score=1000000)

        centipawnloss += sf_score - pr_score
        count += 1
    print (key,centipawnloss/count)
    log_file.write(','.join(map(str,[key,centipawnloss/count]))+'\n')

engine.quit()