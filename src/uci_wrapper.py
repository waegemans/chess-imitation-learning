import chess
import torch
import data_util
from multiprocessing import Pool


board = chess.Board()
model = torch.load("output/model_ep2.nn",map_location=torch.device('cpu'))
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
    state = data_util.board_to_state(board)
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

logfile = open("logfile.log",'w')
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
        go()
    if x.lower() == "quit":
        break
logfile.close()
        
        
        
