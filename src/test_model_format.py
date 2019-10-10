import torch
from models import ssf_asf_1024

model = ssf_asf_1024()
print(model)

data = torch.randn(12,773)
mask = torch.empty(4096).uniform_(0, 1)
mask = torch.bernoulli(mask)

print (mask)
out = model(data,mask)

print(out.shape)
print(out)
