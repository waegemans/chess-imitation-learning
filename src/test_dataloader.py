import dataset

import torch
import time
import progressbar


#pre_ld = torch.utils.data.dataloader.DataLoader(dataset.ChessMoveDataset_pre(), batch_size=1<<12, shuffle=True, num_workers=8)
std_ld = torch.utils.data.dataloader.DataLoader(dataset.ChessMoveDataset(), batch_size=1<<12, shuffle=True, num_workers=8)

start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

for _ in progressbar.progressbar(std_ld):
    continue

end = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

print ("std", end-start)

exit()

start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

for _ in progressbar.progressbar(pre_ld):
    continue

end = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

print ("pre", end-start)
