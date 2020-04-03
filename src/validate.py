import models
import dataset

import numpy as np
import progressbar
import torch
import glob
## 0ab90067a02d8eb69c5aa4756eeed062d4872c5a
ds = dataset.ChessMoveDataset_cp_it(mode='val')
dl = torch.utils.data.DataLoader(ds,batch_size=1024,shuffle=False, num_workers=8, drop_last=True)


device = ('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')

with torch.no_grad():
    for model_file in glob.glob("output/0aff195fdf8987061724c6eebb39bb04099a577e/model_ep66.nn"):
        print(model_file)
        model = torch.load(model_file,map_location=device)
        model.train()
        
        accs = []
        conf_cor = []
        conf_wro = []
        conf = []

        for x,c,m,l in progressbar.progressbar(dl):
            x,c,m,l = x.to(device),c.to(device),m.to(device),l.to(device)

            predicted = torch.stack([model(x).masked_fill(l==0,-float('inf')).argmax(dim=1) for _ in range(10)],dim=1)
            top_label,_ = predicted.mode()
            correct = top_label == c.argmax(dim=1)
            acc = correct.float().mean().item()
            confidence = (predicted.T == top_label).float().mean(dim=0)
            accs.append(acc)
            conf_cor.append(confidence[correct].mean().item())
            conf_wro.append(confidence[~correct].mean().item())
            conf.append(confidence.mean().item())

        print(np.mean(accs),np.mean(conf_cor),np.mean(conf_wro),np.mean(conf))