from copy import deepcopy
import math

import torch
import torch.nn as nn

class ABReLU(nn.Module):
    def __init__(self, rho=1):
        super().__init__()
        b = B(rho)
        self.b = b
    
    def forward(self, x):
        return self.b*torch.minimum(x,torch.tensor(0.))+torch.maximum(x,torch.tensor(0.))

def B(rho):
    b = 0 if rho==1 else (math.sqrt(1-(rho-1)**2)-1)/(rho-1)
    return b

class DenseNet1(nn.Sequential):
    def __init__(self, inp_dim, hdim=1000, rho=1., bias=False):
        lin_1 = nn.Linear(inp_dim, hdim, bias=bias)
        nn.init.normal_(lin_1.weight)
        lin_2 = nn.Linear(hdim, 1, bias=bias)
        nn.init.normal_(lin_2.weight)
        super().__init__(
            lin_1, ABReLU(rho), lin_2
        )
        self.hdim = hdim
    
    def forward(self, x):
        x = math.sqrt(2/self.hdim)*super().forward(x)
        x = torch.squeeze(x, -1)
        return x

class ModelWithNTK(nn.Module):
    def __init__(self):
        super().__init__()
    
    def get_gradient(self, x):
        self.zero_grad()
        y = self(x)
        y.backward()
        return torch.cat([param.grad.flatten() for param in self.parameters()])

    def ntk_features(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        feats = torch.stack([
            self.get_gradient(_x) for _x in x
        ], dim=0)
        return feats.reshape(*shape[:(-1)], -1).detach().clone()

class DenseNet(ModelWithNTK):
    def __init__(self, inp_dim, hdims=None, outp_dim=1, rho=1, bias=False, linear_readout=False, nonlinearity='piecewise_linear'):
        super().__init__()
        hdims = hdims or []
        L = []
        for i, (_in, _out) in enumerate(zip([inp_dim]+hdims[:-1], hdims)):
            linear = nn.Linear(_in, _out, bias=bias)
            if nonlinearity == 'piecewise_linear':
                nn.init.normal_(linear.weight, std=math.sqrt(2/(((B(rho)-1)**2)*_in)))
            elif nonlinearity == 'tanh':
                nn.init.xavier_normal_(linear.weight)
            L.append(linear)
            if nonlinearity == 'piecewise_linear':
                L.append(ABReLU(rho))
            elif nonlinearity == 'tanh':
                L.append(nn.Tanh())
        self.features = nn.Sequential(*L)
        if len(hdims) > 0:
            _in = hdims[-1]
        else:
            _in = inp_dim
        self.readout = nn.Linear(_in, outp_dim, bias=bias)
        if linear_readout:
            nn.init.zeros_(self.readout.weight)
        else:
            if nonlinearity == 'piecewise_linear':
                nn.init.normal_(self.readout.weight, std=math.sqrt(2/(((B(rho)-1)**2)*_out)))
            elif nonlinearity == 'tanh':
                nn.init.xavier_normal_(self.readout.weight)
        self.linear_readout = linear_readout
        self.outp_dim = outp_dim
    
    def parameters(self):
        if self.linear_readout:
            return self.readout.parameters()
        return super().parameters()
    
    def forward(self, x):
        x = self.features(x)
        x = self.readout(x)
        if self.outp_dim == 1:
            x = torch.squeeze(x, -1)
        return x

# Alias for backward compatibility
DenseNet2 = DenseNet

class ZeroOutput(ModelWithNTK):
    def __init__(self, module, scaling=1.):
        super().__init__()
        self.module = module
        self.init_module = deepcopy(module)
        self.scaling = scaling
    
    def parameters(self):
        return self.module.parameters()
    
    def forward(self, x):
        return self.scaling*(self.module(x)-self.init_module(x))

def add_argparse_arguments(parser):
    parser.add_argument('--hdims', nargs='*', default=[], type=int)
    parser.add_argument('--rho', type=float, default=1)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--scaling', default=1., type=float)
    parser.add_argument('--model_seed', default=1, type=int)
    parser.add_argument('--mode', choices=['backprop', 'linear_readout', 'ntk'], default='backprop')
    parser.add_argument('--nonlinearity', choices=['piecewise_linear', 'tanh'], default='piecewise_linear')
    return parser

def get_network(args):
    torch.manual_seed(args.model_seed)
    model = DenseNet(
        inp_dim=2*args.n, hdims=args.hdims, rho=args.rho, bias=args.bias, linear_readout=(args.mode=='linear_readout')
    )
    if args.mode != 'linear_readout':
        model = ZeroOutput(model, scaling=args.scaling)
    return model
