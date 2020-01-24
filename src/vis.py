import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

batches_per_epoch = 139

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

fig,ax = plt.subplots(4)

ax[0].plot(data['batch_count'],data['train_cross_entropy_loss'], color='blue', alpha=0.1)
ax[0].plot(vdata['batch_count'],vdata['val_cross_entropy_loss'], color='orange', alpha=0.1)


ax[1].plot(data['batch_count'],data['train_acc'], color='blue', alpha=0.1)
ax[1].plot(vdata['batch_count'],vdata['val_acc'], color='orange', alpha=0.1)

x,y = smooth_window(data['train_cross_entropy_loss'],data['batch_count'])
ax[0].plot(y,x,label='Train Cross entropy', color='blue')
x,y = smooth_window(vdata['val_cross_entropy_loss'],vdata['batch_count'])
ax[0].plot(y,x,label='Val Cross entropy', color='orange')


x,y = smooth_window(data['train_acc'],data['batch_count'])
ax[1].plot(y,x,label='Training acc', color='blue')
x,y = smooth_window(vdata['val_acc'],vdata['batch_count'])
ax[1].plot(y,x,label='Validation acc', color='orange')

ax[2].plot(data['batch_count'],data['train_grads'], color='blue', alpha=0.1)
x,y = smooth_window(data['train_grads'],data['batch_count'])
ax[2].plot(y,x,label='Training grads', color='blue')

if 'train_min_cp' in data.keys():
    ax[3].plot(data['batch_count'],data['train_min_cp'], color='blue', alpha=0.1)
    ax[3].plot(vdata['batch_count'],vdata['val_min_cp'], color='orange', alpha=0.1)
    
    x,y = smooth_window(data['train_min_cp'],data['batch_count'])
    ax[3].plot(y,x,label='Training min cp loss', color='blue')
    x,y = smooth_window(vdata['val_min_cp'],vdata['batch_count'])
    ax[3].plot(y,x,label='Validation min cp loss', color='orange')


for x in range(0,data['batch_count'].values[-1],batches_per_epoch):
  ax[0].axvline(x=x, ymin=0.0, ymax=1.0, color='r', alpha=0.1)
  ax[1].axvline(x=x, ymin=0.0, ymax=1.0, color='r', alpha=0.1)
  ax[2].axvline(x=x, ymin=0.0, ymax=1.0, color='r', alpha=0.1)
  ax[3].axvline(x=x, ymin=0.0, ymax=1.0, color='r', alpha=0.1)

ax[1].legend()
ax[0].legend()
ax[2].legend()
ax[3].legend()
plt.show()
