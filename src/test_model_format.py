import torch
from models import MultiAttentionHead

m = MultiAttentionHead(64,64,64,64,64,4)

x = torch.randn(5,4,64)

y,z = m(x,x)
print(y.shape)
print(z.shape)

print(z)
