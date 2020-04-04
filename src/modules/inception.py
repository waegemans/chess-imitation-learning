import torch
import torch.nn as nn

class cnn_inception_chess(nn.Module):
    def __init__(self,c_in,c_out):
        super(cnn_inception_chess, self).__init__()
        if c_out % 8 != 0:
            raise ValueError("Number of output channels must be devisable by 8")
        cl_out = c_out // 8

        convs = [nn.Conv2d(c_in,cl_out,kernel_size=3,padding=i,dilation=i) for i in range(1,8)]
        convs.append(nn.Conv2d(c_in,cl_out,kernel_size=5,padding=2))
        
        self.convs = nn.ModuleList(convs)

    def forward(self,x):
        out_list = [m(x) for m in self.convs]
        out = torch.cat(out_list,dim=1)
        return out