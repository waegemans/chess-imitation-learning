import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('output/play_acpl.csv',delimiter=',')

for key,group in data.groupby(['lambda']):
    #ax[0].scatter(group['epoch'],group['cpl_model'],label=key,alpha=0.1)
    df_mean = group.groupby(['epoch']).mean()
    plt.scatter(df_mean.index,df_mean['cpl_model'],label='Model lambda=%f'%key,alpha=1)
    plt.scatter(df_mean.index,df_mean['cpl_rand'],label='Random lambda=%f'%key,alpha=0.5)

    plt.title("Average centipawn loss over epochs (Calculated from Stockfish depth 12)")
    plt.legend()

plt.show()
