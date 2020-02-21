import torch
import numpy as np
import chess
import csv
import glob
import util
import ast
import progressbar

def to_state_value_csv():
    with open("data/depth18_gamma0.200000/moves_shuf.csv", "r") as csv_file_in:
        with open("data/depth18_gamma0.200000/moves_statevalue.csv", "w") as csv_file_out:
            r = csv.reader(csv_file_in)
            w = csv.writer(csv_file_out)

            print('Loading data...')
            for fen_without_count,dstr in progressbar.progressbar(r):
                cpdict = ast.literal_eval(dstr)
                fen = fen_without_count + " 0 1"
                board = chess.Board(fen=fen)
                for uci,value in cpdict.items():
                    board.push_uci(uci)
                    w.writerow([board.fen(),value])
                    board.pop()


class ChessMoveDataset_statevalue_it(torch.utils.data.IterableDataset):
    def __init__(self, mode='train', precompute=False,discretize=False):
        super(ChessMoveDataset_statevalue_it,self).__init__()
        self.mode = mode
        self.discretize = discretize
        self.bins = [-5000,-100,100,5000]
        if precompute:
            self.precompute()
        else:
            self.num_of_splits = len(glob.glob('data/depth18_gamma0.200000/statevalue/pre/cnn_%s_*.npy'%self.mode))
            self.n_items = self.num_of_splits*10000

    
    def precompute(self):
        with open("data/depth18_gamma0.200000/moves_statevalue_%s.csv"%(self.mode), "r") as csv_file:
            r = csv.reader(csv_file)
            self.n_items = 0

            self.cnn = []
            self.values = []
            self.num_of_splits = 0

            print('Loading data...')
            for fen,value in progressbar.progressbar(r):
                board = chess.Board(fen=fen)
                state = util.board_to_state(board)
                cnn = util.state_to_cnn(state)

                self.cnn.append(cnn)
                self.values.append(int(value))
                self.n_items += 1
                if self.n_items % 10000 == 0:
                    self.save_precomputed()
            self.save_precomputed()
        
    def save_precomputed(self):
        if self.cnn is []:
            return
        np.save('data/depth18_gamma0.200000/statevalue/pre/cnn_%s_%d.npy'%(self.mode,self.num_of_splits), np.array(self.cnn))
        np.save('data/depth18_gamma0.200000/statevalue/pre/values_%s_%d.npy'%(self.mode,self.num_of_splits), np.array(self.values))
        self.cnn = []
        self.values = []
        self.num_of_splits += 1



    def __iter__(self):
        it = None
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            it = range(0,self.num_of_splits,1)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            it = range(worker_id, self.num_of_splits, num_workers)
        
        for idx in it:
            cnn = np.load('data/depth18_gamma0.200000/statevalue/pre/cnn_%s_%d.npy'%(self.mode,idx))
            value = np.load('data/depth18_gamma0.200000/statevalue/pre/values_%s_%d.npy'%(self.mode,idx))
            if self.discretize:
                value = np.digitize(value,self.bins)
            else:
                value /= 10000

            for j in range(len(cnn)):
                yield torch.tensor(cnn[j], dtype=torch.float) ,torch.tensor(value[j], dtype=(torch.float,torch.long)[self.discretize])

    def __len__(self):
        return self.n_items
