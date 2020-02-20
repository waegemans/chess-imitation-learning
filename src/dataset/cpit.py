import torch
import numpy as np
import chess
import csv
import glob
import util
import ast
import progressbar


class ChessMoveDataset_cp_it(torch.utils.data.IterableDataset):
    def __init__(self, mode='train', precompute=False):
        super(ChessMoveDataset_cp_it,self).__init__()
        self.mode = mode
        if precompute:
            self.precompute()
        else:
            self.num_of_splits = len(glob.glob('data/depth18_gamma0.200000/pre/cnn_%s_*.npy'%self.mode))
            self.n_items = self.num_of_splits*10000

    
    def precompute(self):
        with open("data/depth18_gamma0.200000/moves_%s.csv"%(self.mode), "r") as csv_file:
            r = csv.reader(csv_file)
            self.n_items = 0

            self.cnn = []
            self.cp_loss = []
            self.mask = []
            self.legal_mask = []
            self.num_of_splits = 0

            print('Loading data...')
            for fen_without_count,dstr in progressbar.progressbar(r):
                cpdict = ast.literal_eval(dstr)
                fen = fen_without_count + " 0 1"
                board = chess.Board(fen=fen)
                state = util.board_to_state(board)
                legal_mask = util.movelist_to_actionmask(board.legal_moves)
                cnn = util.state_to_cnn(state)
                cp_loss,mask = util.cpdict_to_loss_mask(cpdict)

                self.cnn.append(cnn)
                self.cp_loss.append(cp_loss)
                self.mask.append(mask)
                self.legal_mask.append(legal_mask)
                self.n_items += 1
                if self.n_items % 10000 == 0:
                    self.save_precomputed()
            self.save_precomputed()
        
    def save_precomputed(self):
        if self.cnn is []:
            return
        np.save('data/depth18_gamma0.200000/pre/cnn_%s_%d.npy'%(self.mode,self.num_of_splits), np.array(self.cnn))
        np.save('data/depth18_gamma0.200000/pre/cp_loss_%s_%d.npy'%(self.mode,self.num_of_splits), np.array(self.cp_loss))
        np.save('data/depth18_gamma0.200000/pre/mask_%s_%d.npy'%(self.mode,self.num_of_splits), np.array(self.mask))
        np.save('data/depth18_gamma0.200000/pre/legal_mask_%s_%d.npy'%(self.mode,self.num_of_splits), np.array(self.legal_mask))
        self.cnn = []
        self.cp_loss = []
        self.mask = []
        self.legal_mask = []
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
            cnn = np.load('data/depth18_gamma0.200000/pre/cnn_%s_%d.npy'%(self.mode,idx))
            cp_loss = np.load('data/depth18_gamma0.200000/pre/cp_loss_%s_%d.npy'%(self.mode,idx))
            mask = np.load('data/depth18_gamma0.200000/pre/mask_%s_%d.npy'%(self.mode,idx))
            legal_mask = np.load('data/depth18_gamma0.200000/pre/legal_mask_%s_%d.npy'%(self.mode,idx))

            for j in range(len(cnn)):
                yield torch.tensor(cnn[j], dtype=torch.float) ,torch.tensor(cp_loss[j], dtype=torch.float) ,torch.tensor(mask[j], dtype=torch.float), torch.tensor(legal_mask[j], dtype=torch.float)


    def __len__(self):
        return self.n_items