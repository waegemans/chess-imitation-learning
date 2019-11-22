import numpy as np
import chess
import torch
import csv

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