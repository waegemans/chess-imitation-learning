import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

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

if __name__ == "__main__":
  if (len(sys.argv) < 1):
      print("No file provided!")
      exit()
  file_name = sys.argv[1]
  data = pd.read_csv(file_name)
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

  if 'train_1cp_acc' in data.keys():
    ax[1].plot(data['batch_count'],data['train_1cp_acc'], color='cyan', alpha=0.1)
    ax[1].plot(vdata['batch_count'],vdata['val_1cp_acc'], color='salmon', alpha=0.1)

    x,y = smooth_window(data['train_1cp_acc'],data['batch_count'])
    ax[1].plot(y,x,label='Training acc', color='cyan')
    x,y = smooth_window(vdata['val_1cp_acc'],vdata['batch_count'])
    ax[1].plot(y,x,label='Validation acc', color='salmon')


  ax[2].plot(data['batch_count'],data['train_grads'], color='blue', alpha=0.1)
  x,y = smooth_window(data['train_grads'],data['batch_count'])
  ax[2].plot(y,x,label='Training grads', color='blue')

  if 'train_min_cp' in data.keys():
      ax[3].plot(data['batch_count'],data['train_min_cp'], color='blue', alpha=0.1)
      ax[3].plot(vdata['batch_count'],vdata['val_min_cp'], color='orange', alpha=0.1)
      
      x,y = smooth_window(data['train_min_cp'],data['batch_count'])
      ax[3].plot(y,x,label='Training min pawn loss', color='blue')
      x,y = smooth_window(vdata['val_min_cp'],vdata['batch_count'])
      ax[3].plot(y,x,label='Validation min pawn loss', color='orange')

  prev_e = -1
  for e,b in zip(data['epoch'],data['batch_count']):
    if e == prev_e:
      continue
    prev_e = e
    ax[0].axvline(x=b, ymin=0.0, ymax=1.0, color='r', alpha=0.1)
    ax[1].axvline(x=b, ymin=0.0, ymax=1.0, color='r', alpha=0.1)
    ax[2].axvline(x=b, ymin=0.0, ymax=1.0, color='r', alpha=0.1)
    ax[3].axvline(x=b, ymin=0.0, ymax=1.0, color='r', alpha=0.1)

  log = False
  if log:
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[2].set_xscale('log')
    ax[3].set_xscale('log')
  ax[1].legend(prop={'size': 6})
  ax[0].legend(prop={'size': 6})
  ax[2].legend(prop={'size': 6})
  ax[3].legend(prop={'size': 6})
  plt.show()
