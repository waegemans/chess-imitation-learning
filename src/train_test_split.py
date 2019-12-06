import numpy as np
import math

num_of_files = 41

# percentage of validation set
ratio = 0.1

for idx in range(num_of_files):
    b = np.load('data/pre_pov/boards_%d.npy'%(idx))
    m = np.load('data/pre_pov/moves_%d.npy'%(idx))

    num = len(b)

    num_val = int(math.floor(num*ratio))

    val_b = b[:num_val]
    val_m = m[:num_val]

    train_b = b[num_val:]
    train_m = m[num_val:]

    np.save('data/pre_pov/boards_val_%d.npy'%(idx),val_b)
    np.save('data/pre_pov/moves_val_%d.npy'%(idx),val_m)

    np.save('data/pre_pov/boards_train_%d.npy'%(idx),train_b)
    np.save('data/pre_pov/moves_train_%d.npy'%(idx),train_m)
