import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def smooth_exp(x):
  y = np.zeros_like(x, dtype=np.float)
  for i,xi in enumerate(x):
    if i == 0:
      y[i] = xi
    else:
      y[i] = 0.9*y[i-1]+0.1*xi
  return y

data = pd.read_csv('1kpos_80_20split_simple_64bs.csv')

fig,ax = plt.subplots(2)
plt.title("Stockfish 1k positions")

ax[0].plot(data['train_cross_entropy_loss'],label='Train Cross entropy')
ax[0].plot(data['val_cross_entropy_loss'],label='Train Cross entropy')


ax[1].plot(data['train_acc'],label='Training acc')
ax[1].plot(data['val_acc'],label='Validation acc')

ax[1].legend()
ax[0].legend()
plt.show()
