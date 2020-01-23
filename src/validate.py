import models
import dataset

import progressbar
import torch
import glob
## 0ab90067a02d8eb69c5aa4756eeed062d4872c5a
ds = dataset.ChessMoveDataset_pre_it_pov_cnn('val')
dl = torch.utils.data.DataLoader(ds,batch_size=1024,shuffle=False, num_workers=8)


device = ('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')

with torch.no_grad():
    for model_file in glob.glob("output/0ab90067a02d8eb69c5aa4756eeed062d4872c5a/*.nn"):
        print(model_file)
        model = torch.load(model_file,map_location=device)
        model.eval()
        
        val_acc = 0
        n_samples = 0

        for x,y in progressbar.progressbar(dl):
            x,y = x.to(device),y.to(device)

            predicted = model(x)

            n_samples += len(x)
            val_acc += (predicted.detach().argmax(dim=1) == y).cpu().numpy().sum()

        print(model_file, val_acc/n_samples)