import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

batches_per_epoch = 1686

def smooth_exp(x):
  y = np.zeros_like(x, dtype=np.float)
  for i,xi in enumerate(x):
    if i == 0:
      y[i] = xi
    else:
      y[i] = 0.9*y[i-1]+0.1*xi
  return y

def smooth_window(x,size=7):
  y = np.convolve(x,np.ones(size),'same')
  n = size//2
  y[:n] /= size - np.arange(n,0,-1)
  y[-n:] /= size - np.arange(1,n+1)
  y[n:-n] /= size
  return y

data = pd.read_csv('output/tmp.csv')
vdata = data.dropna()

fig,ax = plt.subplots(2)

ax[0].plot(data['batch_count'],data['train_cross_entropy_loss'], color='blue', alpha=0.1)
ax[0].plot(vdata['batch_count'],vdata['val_cross_entropy_loss'], color='orange', alpha=0.1)


ax[1].plot(data['batch_count'],data['train_acc'], color='blue', alpha=0.1)
ax[1].plot(vdata['batch_count'],vdata['val_acc'], color='orange', alpha=0.1)

ax[0].plot(data['batch_count'],smooth_window(data['train_cross_entropy_loss']),label='Train Cross entropy', color='blue')
ax[0].plot(vdata['batch_count'],smooth_window(vdata['val_cross_entropy_loss']),label='Val Cross entropy', color='orange')


ax[1].plot(data['batch_count'],smooth_window(data['train_acc']),label='Training acc', color='blue')
ax[1].plot(vdata['batch_count'],smooth_window(vdata['val_acc']),label='Validation acc', color='orange')

for x in range(0,data['batch_count'].values[-1]+batches_per_epoch,batches_per_epoch):
  ax[0].axvline(x=x, ymin=0.0, ymax=1.0, color='r', alpha=0.1)
  ax[1].axvline(x=x, ymin=0.0, ymax=1.0, color='r', alpha=0.1)

ax[1].legend()
ax[0].legend()
plt.show()
