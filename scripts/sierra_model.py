## outlines model specs

import torch
import torch.nn as nn

'''
class EWTModel(nn.Module):
    def __init__(self, n_features=7, hidden_dims=[64, 32, 16]):
        super().__init__()
        layers = []
        in_dim = n_features
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            # LeakyReLU with alpha=0.4 to allow small gradients for negative inputs
            layers.append(nn.LeakyReLU(0.4)) 
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # clean squeeze for 1D target (Batch,)
        return self.net(x).squeeze(-1)
'''

class EWTModel(nn.Module):
    def __init__(self, n_features, hidden_dims=[64, 32, 16], dropout_rate=0.2):
        super().__init__()
        layers = []
        in_dim = n_features
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim)) 
            layers.append(nn.LeakyReLU(0.2)) 
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


