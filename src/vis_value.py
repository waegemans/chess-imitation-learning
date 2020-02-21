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

def smooth_window(x,lab,size=51):
  y = np.convolve(x,np.ones(size),'valid')
  n = size//2
  if len(x) > size:
    return y/size,lab[n:-n]
  else:
    return [],[]

def smooth(x):
  return smooth_exp(x)

data = pd.read_csv('output/tmp.csv')
vdata = data.dropna()

fig,ax = plt.subplots(2)

ax[0].plot(data['batch_count'],data['train_mse_loss'], color='blue', alpha=0.1)
ax[0].plot(vdata['batch_count'],vdata['val_mse_loss'], color='orange', alpha=0.1)

x,y = smooth_window(data['train_mse_loss'],data['batch_count'])
ax[0].plot(y,x,label='Train Cross entropy', color='blue')
x,y = smooth_window(vdata['val_mse_loss'],vdata['batch_count'])
ax[0].plot(y,x,label='Val Cross entropy', color='orange')

ax[1].plot(data['batch_count'],data['train_grads'], color='blue', alpha=0.1)
x,y = smooth_window(data['train_grads'],data['batch_count'])
ax[1].plot(y,x,label='Training grads', color='blue')

prev_e = -1
for e,b in zip(data['epoch'],data['batch_count']):
  if e == prev_e:
    continue
  prev_e = e
  ax[0].axvline(x=b, ymin=0.0, ymax=1.0, color='r', alpha=0.1)
  ax[1].axvline(x=b, ymin=0.0, ymax=1.0, color='r', alpha=0.1)
#ax[0].set_yscale('log')
ax[1].legend(prop={'size': 6})
ax[0].legend(prop={'size': 6})
plt.show()
