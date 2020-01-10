import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("output/centipawnloss_agg.csv")


print(data.keys())

plt.title('Centipawnloss on validation set per epoch')
plt.plot(data['epoch'], data['loss_agg'],label='mean centipawnloss')
plt.show()
