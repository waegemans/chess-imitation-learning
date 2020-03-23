import pandas as pd
import sys

if __name__ == "__main__":
    if (len(sys.argv) < 1):
        print("No file provided!")
        exit()
    file_name = sys.argv[1]
    data = pd.read_csv(file_name)

    gdata = data.groupby(['epoch']).mean()[:-1]
    midx = gdata.idxmax()
    sdata = gdata.iloc[midx].copy()
    sdata['field'] = midx.index
    print(sdata.reset_index().set_index('field').loc[['val_acc','val_1cp_acc','val_min_cp']])