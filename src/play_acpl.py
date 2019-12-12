import numpy as np

import chess
import chess.engine

import data_util

import models
import torch

engine = chess.engine.SimpleEngine.popen_uci('../engine/stockfish_10_x64')

log_file = open("output/play_acpl.csv", "w")
log_file.write('epoch,lambda,cpl_model,cpl_rand,mate_lost_model,mate_lost_rand\n')

games_per_setting = 5
epochs = 6
#chance for random play
lambdas = [0,0.5,1]
stockfish_depth = 12


def push_uci_queen(board,uci):
    try:
        board.push_uci(uci)
    except:
        #print("Defaulting to queen promotion")
        board.push_uci(uci+'q')

for e in range(7,8,1):#epochs):
    print("Epoch %d of %d"%(e,epochs))
    model = torch.load('output/model_ep%d.nn'%e)
    model.eval()
    for l in lambdas:
        print("  Lambda %f in "%(l)+str(lambdas))
        for game in range(games_per_setting):
            log_file.flush()
            print("    Game %d of %d"%(game,games_per_setting))
            board = chess.Board()

            while not board.is_game_over():
                if not board.turn:
                    board = board.mirror()
                color = board.turn
                state = data_util.board_to_state(board)
                y = model(torch.tensor(state,dtype=torch.float).unsqueeze(0))[0]
                y -= y.min()
                action = y * torch.tensor(data_util.movelist_to_actionmask(board.legal_moves),dtype=torch.float)
                pred_uci = data_util.action_to_uci(action.detach().data)
                
                pred_eng = engine.play(board, limit=chess.engine.Limit(depth=stockfish_depth+1))
                board.push(pred_eng.move)

                analy_sf = engine.analyse(board, limit=chess.engine.Limit(depth=stockfish_depth))
                board.pop()

                board.push(np.random.permutation(list(board.legal_moves))[0])
                analy_rnd = engine.analyse(board, limit=chess.engine.Limit(depth=stockfish_depth))
                board.pop() 

                push_uci_queen(board,pred_uci)
                analy_model = engine.analyse(board, limit=chess.engine.Limit(depth=stockfish_depth))

                cp_sf = analy_sf['score'].pov(color).score()
                cp_rnd = analy_rnd['score'].pov(color).score()
                cp_model = analy_model['score'].pov(color).score()

                if cp_sf is None:
                    log_file.write(','.join(map(str,[e,l,0,0,int(cp_model is not None),int(cp_rnd is not None)]))+'\n')
                elif cp_model is None or cp_rnd is None:
                    pass
                else:
                    log_file.write(','.join(map(str,[e,l,cp_sf-cp_model,cp_sf-cp_rnd,0,0]))+'\n')

                if np.random.binomial(1,l) == 1:
                    board.pop()
                    board.push(np.random.permutation(list(board.legal_moves))[0])

log_file.close()
engine.quit()
