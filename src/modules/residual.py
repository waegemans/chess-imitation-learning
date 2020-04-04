import torch
import torch.nn as nn
from .inception import cnn_inception_chess

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
    def __init__(self, hidden_channels, kernel_size, dropout_p=0):
        super(cnn_res_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(hidden_channels),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(hidden_channels),
            nn.Dropout2d(p=dropout_p),
        )
    
    def forward(self, x):
        out = self.block(x)
        return nn.functional.relu(out+x)

class inception_res_block(nn.Module):
    def __init__(self, hidden_channels,dropout_p=0):
        super(inception_res_block, self).__init__()
        self.block = nn.Sequential(
            cnn_inception_chess(hidden_channels,hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(),


            cnn_inception_chess(hidden_channels,hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.Dropout2d(p=dropout_p)
        )
    def forward(self, x):
        out = self.block(x)
        return nn.functional.relu(out+x)