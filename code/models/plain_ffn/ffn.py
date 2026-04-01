import torch
import torch.nn as nn


class PlainFFNet(nn.Module):
    def __init__(self, input_dim=100, hidden_dims=[100, 100], output_dim=10, seed=0):
        super(PlainFFNet, self).__init__()
        torch.manual_seed(seed)
        
        hidden_dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(hidden_dims)-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
