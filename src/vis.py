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

data = pd.read_csv('1000_nll_4move.csv')

fig,ax = plt.subplots(2)
plt.title("Stockfish opening 4 moves - batchsize=1")

ax[0].plot(data['mse_loss'],label='NLL', alpha=0.2)
ax[0].plot(np.convolve(data['mse_loss'], np.ones(9)/9, 'same'), label='NLL smooth')

ax[1].plot(smooth_exp(data.stockfish == data.predicted), label='train_accuracy')
'''
ax[1].plot(np.cumsum(data.stockfish == 'e2e4')/(data.sample_nr//2+1), label='e2e4_stockfish')
ax[1].plot(np.cumsum(data.stockfish == 'c2c4')/(data.sample_nr//2+1), label='c2c4_stockfish')
ax[1].plot(np.cumsum(data.stockfish == 'd2d4')/(data.sample_nr//2+1), label='d2d4_stockfish')
ax[1].plot(np.cumsum(data.stockfish == 'e2e3')/(data.sample_nr//2+1), label='e2e3_stockfish')
ax[1].plot(np.cumsum(data.stockfish == 'g1f3')/(data.sample_nr//2+1), label='g1f3_stockfish')
'''
ax[1].legend()
ax[0].legend()
plt.show()
