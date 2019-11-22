import dataset
import numpy as np
import torch


ds = dataset.ChessMoveDataset()

boards,moves = None,None
dataloader = torch.utils.data.dataloader.DataLoader(ds, batch_size=10000, num_workers=8)
j = 0
for i,x in enumerate(dataloader):
    j = i
    a,b = x
    if boards is None:
        boards = a.numpy().astype(np.bool)
    else:
        boards = np.concatenate((boards, a.numpy().astype(np.bool)))

    if moves is None:
        moves = b.numpy().argmax(axis=1)
    else:
        moves = np.concatenate((moves, b.numpy().argmax(axis=1)))

    print(i)
    if (i+1)%64 == 0:
        np.save('data/pre/boards_%d'%((i)//64),boards)
        np.save('data/pre/moves_%d'%((i)//64),moves)
        boards = None
        moves = None

if boards is not None:
    np.save('data/pre/boards_%d'%((i)//64),boards)
if moves is not None:
    np.save('data/pre/moves_%d'%((i)//64),moves)