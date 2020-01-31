import chess
import torch
import data_util
from multiprocessing import Pool
import numpy as np
import time
import sys
import models
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('sample_type', type=str,
                   help='how to choose random move', default='model')
args = parser.parse_args()


device = ('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')

logfile = open("logfile.log",'w')
board = chess.Board()



#model = torch.load("output/model.nn",map_location=device)
model = models.cnn_bare()
model.to(device)
model.eval()

def uci():
    print("id name imitation-net")
    print("id author waegemans")
    print("uciok")
    
def isready():
    print("readyok")

def position(args):
    global board
    if len(args) > 0 and args[0] == "startpos":
        board = chess.Board()
        args = args[1:]
    if len(args) > 0 and args[0] == "moves":
        for m in args[1:]:
            board.push_uci(m)
            args = []
    if len(args) > 0:
        board.board(fen=' '.join(args))
    
def go():
    global board
    b = board
    if not board.turn:
        b = board.mirror()
    state = data_util.board_to_state(b)
    cnn = data_util.state_to_cnn(state)

    y = model(torch.tensor(cnn,dtype=torch.float).unsqueeze(0)).detach()
    y = y - y.min()
    y_masked = y.numpy() * data_util.movelist_to_actionmask(b.legal_moves)
    uci = data_util.action_to_uci(y_masked)

    if not board.turn:
        uci = data_util.flip_uci(uci)
        
    try:
        board.push_uci(uci)
        board.pop()
    except:
        uci += 'q'
    
    print("bestmove " + uci)

def random_move(board,p_dict):
    if args.sample_type == 'random':
        return np.random.choice(list(board.legal_moves)).uci()
    b = board
    p = None
    fen_without_count = ' '.join(board.fen().split()[:-2])
    if fen_without_count in p_dict.keys():
        p = p_dict[fen_without_count]
    else:
        if not board.turn:
            b = board.mirror()
        state = data_util.board_to_state(b)
        cnn = data_util.state_to_cnn(state)

        y = model(torch.tensor(cnn,dtype=torch.float,device=device).unsqueeze(0)).detach()
        y_masked = torch.nn.functional.softmax(y, dim=1) * torch.tensor(data_util.movelist_to_actionmask(b.legal_moves),dtype=torch.float,device=device)
        y_masked = torch.nn.functional.normalize(y_masked,p=1)

        p=y_masked.cpu().numpy().reshape((-1))
        p_dict[fen_without_count] = p
    idx = np.random.choice(64*64, p=p)

    uci = data_util.idx_to_uci(idx)

    if not board.turn:
        uci = data_util.flip_uci(uci)

    try:
        board.push_uci(uci)
        board.pop()
    except:
        uci += np.random.choice(['q','r','b','n'])
    
    return uci


def go_mcts():
    p_dict = {}
    global board
    d = {}
    total_games = 0

    start = time.time()

    while total_games < 20:
        b = board.copy()
        first_uci = random_move(b,p_dict)
        b.push_uci(first_uci)
        move_count = 0

        while not b.is_game_over():
            uci = random_move(b,p_dict)
            b.push_uci(uci)
            move_count += 1
        
        res = b.result()

        if first_uci not in d.keys():
            d[first_uci] = (0,0,0)
        
        wins,draws,losses = d[first_uci]
        total_games += 1

        if res == '1-0':
            if board.turn:
                wins += 1
            else:
                losses += 1
        elif res == '0-1':
            if board.turn:
                losses += 1
            else:
                wins += 1
        else:
            draws += 1
        
        d[first_uci] = wins,draws,losses
    logfile.write("Total Games: %d\n"%total_games)
    best_uci = ''
    min_loss_prob = 1
    best_win_prob = 0
    best_games = 0
    
    for uci in d.keys():
        wins,draws,losses = d[uci]
        games = wins+draws+losses
        if games == 0:
            continue
        loss_prob = losses/games
        win_prob = wins/games

        if loss_prob < min_loss_prob or (loss_prob == min_loss_prob and win_prob > best_win_prob) or (loss_prob == min_loss_prob and win_prob == best_win_prob and games > best_games):
            best_uci = uci
            min_loss_prob = loss_prob
            best_win_prob = win_prob
            best_games = games

    print("bestmove " + best_uci)



#pool = Pool(processes=8)
while True:
    x = input()
    logfile.write(x+'\n')
    logfile.flush()
    if x.lower() == "uci":
        #pool.apply_async(uci)
        uci()
    if x.lower() == "isready":
        #pool.apply_async(isready)
        isready()
    if x.split()[0].lower() == "position":
        position(x.split()[1:])
    if x.split()[0].lower() == "go":
        go_mcts()
    if x.lower() == "quit":
        break
logfile.close()
        
        
        
