import numpy as np
import chess
import torch
import csv
import ast
import progressbar
import glob

import data_util

class ChessMoveDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(ChessMoveDataset,self).__init__()
        with open("data/moves.csv", "r") as csv_file:
            r = csv.reader(csv_file)
            data = []
            for fen,move in r:
                data.append((fen,move))
            self.data = np.array(data)

    def __getitem__(self, idx):
        fen_without_count,uci = self.data[idx]
        fen = fen_without_count + " 0 1"
        b = chess.Board(fen=fen)
        m = chess.Move.from_uci(uci)
        return torch.tensor(data_util.board_to_state(b), dtype=torch.float) ,torch.tensor(data_util.move_to_action(m), dtype=torch.float)

    def __len__(self):
        return len(self.data)

class ChessMoveDataset_cp(torch.utils.data.Dataset):
    def __init__(self):
        super(ChessMoveDataset_cp,self).__init__()
        with open("data/depth18_gamma0.200000/moves.csv", "r") as csv_file:
            r = csv.reader(csv_file)
            data = []
            print('Loading data...')
            for fen_without_count,dstr in progressbar.progressbar(r):
                cpdict = ast.literal_eval(dstr)
                fen = fen_without_count + " 0 1"
                state = data_util.board_to_state(chess.Board(fen=fen))
                cnn = data_util.state_to_cnn(state)
                cp_loss,mask = data_util.cpdict_to_loss_mask(cpdict)
                data.append((cnn,cp_loss,mask))
            self.data = np.array(data)

    def __getitem__(self, idx):
        cnn,cp_loss,mask = self.data[idx]

        return torch.tensor(cnn, dtype=torch.float), torch.tensor(cp_loss, dtype=torch.float), torch.tensor(mask, dtype=torch.float)

    def __len__(self):
        return len(self.data)

def convert_cp_it(fen_without_count, dstr,queue):
    cpdict = ast.literal_eval(dstr)
    fen = fen_without_count + " 0 1"
    state = data_util.board_to_state(chess.Board(fen=fen))
    cnn = data_util.state_to_cnn(state)
    cp_loss,mask = data_util.cpdict_to_loss_mask(cpdict)
    return cnn,cp_loss,mask

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
            self.num_of_splits = 0

            print('Loading data...')
            fen_arr = []
            dstr_arr = []
            for fen_without_count,dstr in progressbar.progressbar(r):
                cpdict = ast.literal_eval(dstr)
                fen = fen_without_count + " 0 1"
                state = data_util.board_to_state(chess.Board(fen=fen))
                cnn = data_util.state_to_cnn(state)
                cp_loss,mask = data_util.cpdict_to_loss_mask(cpdict)

                self.cnn.append(cnn)
                self.cp_loss.append(cp_loss)
                self.mask.append(mask)
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
        self.cnn = []
        self.cp_loss = []
        self.mask = []
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

            for j in range(len(cnn)):
                yield torch.tensor(cnn[j], dtype=torch.float) ,torch.tensor(cp_loss[j], dtype=torch.float) ,torch.tensor(mask[j], dtype=torch.float)


    def __len__(self):
        return self.n_items

class ChessMoveDataset_it(torch.utils.data.IterableDataset):
    def __init__(self):
        super(ChessMoveDataset_it,self).__init__()
        self.num_of_splits = 9

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
            with open("data/split/moves_shuffled_pov%d"%idx, "r") as csv_file:
                r = csv.reader(csv_file)
                for fen_without_count,move in r:
                    b = chess.Board(fen=fen_without_count+" 0 1")
                    m = chess.Move.from_uci(move)
                    yield torch.tensor(data_util.board_to_state(b), dtype=torch.float) ,torch.tensor(data_util.move_to_action(m), dtype=torch.long)


class ChessMoveDataset_pre_it(torch.utils.data.IterableDataset):
    def __init__(self, mode='train'):
        super(ChessMoveDataset_pre_it,self).__init__()
        self.num_of_splits = 41
        self.mode = mode

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
            b = np.load('data/pre/boards_%s_%d.npy'%(self.mode,idx))
            m = np.load('data/pre/moves_%s_%d.npy'%(self.mode,idx))

            for j in range(len(b)):
                yield torch.tensor(b[j], dtype=torch.float) ,torch.tensor(m[j], dtype=torch.long)


class ChessMoveDataset_pre_it_pov(torch.utils.data.IterableDataset):
    def __init__(self, mode='train'):
        super(ChessMoveDataset_pre_it_pov,self).__init__()
        self.num_of_splits = 41
        self.mode = mode

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
            b = np.load('data/pre_pov/boards_%s_%d.npy'%(self.mode,idx))
            m = np.load('data/pre_pov/moves_%s_%d.npy'%(self.mode,idx))

            for j in range(len(b)):
                yield torch.tensor(b[j], dtype=torch.float) ,torch.tensor(m[j], dtype=torch.long)

class ChessMoveDataset_pre_it_pov_cnn(torch.utils.data.IterableDataset):
    def __init__(self, mode='train'):
        super(ChessMoveDataset_pre_it_pov_cnn,self).__init__()
        self.num_of_splits = 41
        self.mode = mode

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
            b = np.load('data/pre_pov/boards_%s_%d.npy'%(self.mode,idx))
            m = np.load('data/pre_pov/moves_%s_%d.npy'%(self.mode,idx))

            for j in range(len(b)):
                yield torch.tensor(data_util.state_to_cnn(b[j]),dtype=torch.float) ,torch.tensor(m[j], dtype=torch.long)

class ChessMoveDataset_pre(torch.utils.data.Dataset):
    def __init__(self):
        super(ChessMoveDataset_pre,self).__init__()
        boards = None
        moves = None
        print("Loading dataset...")
        for i in range(14):
            b = np.load('data/pre/boards_%d.npy'%(i))
            m = np.load('data/pre/moves_%d.npy'%(i))
            if boards is None:
                boards = b
            else:
                boards = np.concatenate((boards, b))

            if moves is None:
                moves = m
            else:
                moves = np.concatenate((moves, m))
        self.boards = boards
        self.moves = moves
        print("Done!")


    def __getitem__(self, idx):
        return torch.tensor(self.boards[idx], dtype=torch.float) ,torch.tensor(self.moves[idx], dtype=torch.long)

    def __len__(self):
        return 8349558