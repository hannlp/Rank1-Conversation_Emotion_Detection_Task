import torch
import torch.nn as nn

class Highway(nn.Module):
    def __init__(self, in_features, out_features, func, n_layers=1, bias=True):
        super().__init__()
        self.n_layers = n_layers
        self.func = func
        self.nonlinear = nn.ModuleList([nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
                                        for _ in range(n_layers)])
        self.linear = nn.ModuleList([nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
                                        for _ in range(n_layers)])
        self.gate = nn.ModuleList([nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
                                        for _ in range(n_layers)])

    def forward(self, x):
        for i in range(self.n_layers):
            gate = torch.sigmoid(self.gate[i](x))
            linear = self.linear[i](x)
            nonlinear = self.func(self.nonlinear[i](x))
            x = gate * nonlinear + (1 - gate) * linear
        return x