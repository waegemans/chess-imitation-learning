import torch
from models import Att_small

m = Att_small()

x = torch.randn(3,2,71)
y = torch.randn(3,4,128)

o =m(x,y)
print(o.shape)
