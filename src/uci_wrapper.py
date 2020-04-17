import chess
import torch
import util
from multiprocessing import Pool
import numpy as np
import time
import sys
import models
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
                   help='model file')
parser.add_argument('--model_type', type=str,
                   default='siam')
args = parser.parse_args()


device = ('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')

logfile = open("logfile.log",'w')
board = chess.Board()

n_par_games = 64


model = torch.load(args.model,map_location=device)
#model = models.cnn_bare()
#model.to(device)
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
        board = chess.Board(fen=' '.join(args))
    
def go():
    global board
    b = board
    if not board.turn:
        b = board.mirror()
    state = util.board_to_state(b)
    cnn = util.state_to_cnn(state)
    
    model.train()
    with torch.no_grad():
        d = {}
        for _ in range(10):
            y = model(torch.tensor(cnn,dtype=torch.float,device=device).unsqueeze(0)).detach()
            y = y - y.min()
            y_masked = y.cpu().numpy() * util.movelist_to_actionmask(b.legal_moves)
            uci = util.action_to_uci(y_masked)

            if not board.turn:
                uci = util.flip_uci(uci)
                
            try:
                board.push_uci(uci)
                board.pop()
            except:
                uci += 'q'
            if uci not in d.keys():
                d[uci] = 1
            else:
                d[uci] += 1

        maxcount,bestuci=0,None
        for uci,count in d.items():
            if maxcount < count:
                bestuci = uci
                maxcount = count
        print("bestmove " + uci)

def go_state(mode='buckets'):
    global board
    b = board
    if not board.turn:
        b = board.mirror()

    moves = list(b.legal_moves)
    cnn = []
    for mv in moves:
        b.push(mv)
        state = util.board_to_state(b)
        cnn.append(util.state_to_cnn(state))
        b.pop()
    model.train()
    with torch.no_grad():
        y = model(torch.tensor(cnn,dtype=torch.float,device=device)).detach()
        idx = 0
        if mode is 'buckets':
            idx = y.argmax(dim=1).argmax(dim=0).cpu().numpy()
        elif mode is 'value':
            idx = y.argmax().cpu().numpy()

        uci = moves[idx].uci()

        if not board.turn:
            uci = util.flip_uci(uci)
        
        print("bestmove " + uci)

def random_move(boards,p_dict):
    if args.sample_type == 'random':
        return [np.random.choice(list(b.legal_moves)).uci() for b in boards]
    cnn_t = torch.zeros(n_par_games,17,8,8,dtype=torch.float,device=device)
    mask_t = torch.zeros(n_par_games,64*64,dtype=torch.float,device=device)
    fen_without_count = [None for _ in boards]
    for i,b in enumerate(boards):
        turn = b.turn
        #p = None
        #fen_without_count[i] = ' '.join(board.fen().split()[:-2])
        #if fen_without_count[i] in p_dict.keys():
        #    p = p_dict[fen_without_count[i]]
        if not turn:
            b = b.mirror()
        state = util.board_to_state(b)
        cnn = util.state_to_cnn(state)
        cnn_t[i] = torch.tensor(cnn).to(device)
        mask_t[i] = torch.tensor(util.movelist_to_actionmask(b.legal_moves),dtype=torch.float,device=device)

    y = model(cnn_t).detach()
    y_masked = torch.nn.functional.softmax(y, dim=1) * mask_t
    y_masked = torch.nn.functional.normalize(y_masked,p=1)

    p=y_masked.cpu().numpy()
    
    ucis = []
    for i,b in enumerate(boards):
        #p_dict[fen_without_count[i]] = p[i]
        idx = np.random.choice(64*64, p=p[i])

        uci = util.idx_to_uci(idx)

        if not b.turn:
            uci = util.flip_uci(uci)

        try:
            b.push_uci(uci)
            b.pop()
        except:
            uci += np.random.choice(['q','r','b','n'])
        
        ucis.append(uci)
    return ucis


def go_mcts():
    p_dict = {}
    global board
    d = {}
    total_games = 0

    start = time.time()

    boards = [board.copy() for _ in range(n_par_games)]
    first_ucis = [None for _ in range(n_par_games)]
    
    while time.time() - start < 3:
        for i in range(n_par_games):
            b,first_uci = boards[i],first_ucis[i]
            if b.is_game_over():
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
                boards[i] = board.copy()
                first_ucis[i] = None
            
            #
        next_ucis = random_move(boards,p_dict)
        for i in range(n_par_games):
            boards[i].push_uci(next_ucis[i])
            if first_ucis[i] is None:
                first_ucis[i] = next_ucis[i]
            
                
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
    if best_uci == '':
        best_uci = np.random.choice(list(board.legal_moves)).uci()

    print("bestmove " + best_uci)


def go_cmp(binary=False):
    b = board
    if not b.turn :
        b = b.mirror()
    
    legal_moves = list(b.legal_moves)
    best_move = legal_moves[0]
    
    for _ in range(5):
        for mv in legal_moves:
            if mv is best_move:
                continue
            b.push(best_move)
            best = util.state_to_cnn(util.board_to_state(b))
            b.pop()
            b.push(mv)
            comp = util.state_to_cnn(util.board_to_state(b))
            b.pop()
            with torch.no_grad():
                model.train()
                t = torch.tensor(np.array([best,comp]),dtype=torch.float).to(device)
                x = model(t).detach().cpu()
                
                if not binary and x[0].argmax() < 10:
                    best_move = mv
                elif binary and x[0] > 0:
                    best_move = mv
    uci = best_move.uci()

    if not board.turn:
        uci = util.flip_uci(uci)
        
    print("bestmove " + uci)


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
        if args.model_type == 'siam':
            go_cmp()
        if args.model_type == 'siambinary':
            go_cmp(True)
        elif args.model_type == 'buckets':
            go_state(mode='buckets')
        elif args.model_type == 'value':
            go_state(mode='value')
        else:
            go()
    if x.lower() == "quit":
        break
logfile.close()
        
        
        
