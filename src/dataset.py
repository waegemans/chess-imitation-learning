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
            self.data = []
            for fen,move in r:
                self.data.append((fen,move))

    def __getitem__(self, idx):
        fen_without_count,uci = self.data[idx]
        fen = fen_without_count + " 0 1"
        b = chess.Board(fen=fen)
        m = chess.Move.from_uci(uci)
        return data_util.board_to_state(b) ,data_util.move_to_action(m)

    def __len__(self):
        return len(self.data)