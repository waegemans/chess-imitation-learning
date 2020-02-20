import torch
import torch.nn as nn

class fc_res_block(nn.Module):
    def __init__(self, hidden_size):
        super(fc_res_block,self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            )
        
    def forward(self, x):
        out = self.block(x)
        return nn.functional.relu(out+x)

class cnn_res_block(nn.Module):
    def __init__(self, hidden_channels, kernel_size):
        super(cnn_res_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(hidden_channels)
        )
    
    def forward(self, x):
        out = self.block(x)
        return nn.functional.relu(out+x)