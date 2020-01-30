import torch
import numpy as np
import chess
import chess.engine

import models
import data_util

engine = chess.engine.SimpleEngine.popen_uci('../engine/stockfish_10_x64')

model = torch.load('output/model.nn')
model.eval()


centiloss_sum = 0
missed_mate = 0
found_mate = 0
mates = 0
count = 0

for i in range(20):
    print('Game',i)
    board = chess.Board()
    analyze = engine.analyse(board,limit=chess.engine.Limit(depth=20))
    prev_centi = analyze['score'].pov(board.turn)
    while not board.is_game_over():
        state = data_util.board_to_state(board)
        pred = model(torch.tensor(state,dtype=torch.float32))
        legal_mask = data_util.movelist_to_actionmask(board.legal_moves)
        puci = data_util.action_to_uci(pred.detach().numpy()*legal_mask)
        try:
            board.push_uci(puci)
        except:
            board.push_uci(puci+'q')
    
        analyze = engine.analyse(board,limit=chess.engine.Limit(depth=20))
        this_centi = analyze['score'].pov(not board.turn)
        
        if prev_centi.score() is None:
            mates += 1
            if this_centi.score() is not None:
                missed_mate += 1
        elif this_centi.score() is None:
            found_mate += 1
        else:
            centiloss_sum += prev_centi.score() - this_centi.score()
            count += 1
        
        #prev_centi = -this_centi
        # random play
        board.pop()
        board.push(np.random.permutation(list(board.legal_moves))[0])
        
        analyze = engine.analyse(board,limit=chess.engine.Limit(depth=10))
        prev_centi = analyze['score'].pov(board.turn)
  
  
print('avg centiloss: ',centiloss_sum/count)
print('mates_missed: ',missed_mate/mates)
print('mates_found: ',found_mate)
engine.quit()
  
