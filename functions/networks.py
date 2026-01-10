import math
from copy import deepcopy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics as tm
from torch.distributions.normal import Normal

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
    
    def parameters(self):
        if self.linear_readout:
            return self.readout.parameters()
        return super().parameters()
    
    def forward(self, x):
        x = self.features(x)
        x = self.readout(x)
        return x

class MTLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, weight_scaling=None) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.weight_scaling = weight_scaling
    
    def forward(self, input, task):
        bias = None if self.bias is None else self.weight_scaling[task]*self.bias
        return F.linear(input, self.weight_scaling[task]*self.weight, bias)
    
class MTSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
    
    def forward(self, input, task):
        for module in self:
            if isinstance(module, (MTLinear,)):
                input = module(input, task)
            else:
                input = module(input)
        return input
    
class MTDenseNet(ModelWithNTK):
    def __init__(self, inp_dim, hdims=None, outp_dim=1, rho=1, bias=False,
                 linear_readout=False, nonlinearity='piecewise_linear', weight_scaling=None):
        super().__init__()
        hdims = hdims or []
        L = []
        for i, (_in, _out) in enumerate(zip([inp_dim]+hdims[:-1], hdims)):
            linear = MTLinear(_in, _out, bias=bias, weight_scaling=weight_scaling)
            if nonlinearity == 'piecewise_linear':
                nn.init.normal_(linear.weight, std=math.sqrt(2/(((B(rho)-1)**2)*_in)))
            elif nonlinearity == 'tanh':
                nn.init.xavier_normal_(linear.weight)
            L.append(linear)
            if nonlinearity == 'piecewise_linear':
                L.append(ABReLU(rho))
            elif nonlinearity == 'tanh':
                L.append(nn.Tanh())
        self.features = MTSequential(*L)
        if len(hdims) > 0:
            _in = hdims[-1]
        else:
            _in = inp_dim
        self.readout = MTLinear(_in, outp_dim, bias=bias, weight_scaling=weight_scaling)
        if linear_readout:
            nn.init.zeros_(self.readout.weight)
        else:
            if nonlinearity == 'piecewise_linear':
                nn.init.normal_(self.readout.weight, std=math.sqrt(2/(((B(rho)-1)**2)*_out)))
            elif nonlinearity == 'tanh':
                nn.init.xavier_normal_(self.readout.weight)
        self.linear_readout = linear_readout
    
    def parameters(self):
        if self.linear_readout:
            return self.readout.parameters()
        return super().parameters()
    
    def forward(self, x, task):
        x = self.features(x, task)
        x = self.readout(x, task)
        return x
    
class DiagonalReLUNetwork(ModelWithNTK):
    def __init__(self, inp_dim, hdim=1000, outp_dim=1, scaling=1.):
        super().__init__()
        self.dense = nn.Linear(inp_dim, hdim, bias=False)
        weight = Normal(0, 1).sample((hdim, inp_dim))
        self.dense.weight = nn.Parameter(weight/torch.sqrt((weight**2).sum(dim=1, keepdim=True)))
        self.d1 = nn.Parameter(torch.sign(torch.rand(hdim)-0.5)*torch.pow(2/torch.tensor(hdim), 1/4))
        self.d2 = nn.Linear(hdim, outp_dim, bias=False)
        self.d2.weight = nn.Parameter(torch.sign(torch.rand((outp_dim,hdim))-0.5)*torch.pow(2/torch.tensor(hdim), 1/4))
        self.scaling = scaling
    
    def forward(self, x):
        outp = self.dense(x)
        outp = torch.relu(outp)
        outp = self.d1*outp
        outp = self.d2(outp)
        outp = self.scaling*outp
        return outp
    
    def parameters(self):
        return [self.d1] + list(self.d2.parameters())

class ZeroOutput(ModelWithNTK):
    def __init__(self, module, scaling=1., subtract=True):
        super().__init__()
        self.module = module
        if subtract:
            self.init_module = deepcopy(module)
        self.scaling = scaling
        self.subtract = subtract
    
    def parameters(self):
        return self.module.parameters()
    
    def forward(self, x):
        if self.subtract:
            with torch.no_grad():
                init_outp = self.init_module(x)
            return self.scaling*(self.module(x)-init_outp)
        else:
            return self.scaling*self.module(x)
    
class NetworkModule(ZeroOutput):
    def __init__(self, hparams=None, **kwargs):
        if hparams is None:
            parser = argparse.ArgumentParser()
            parser = NetworkModule.add_args(parser)
            hparams = parser.parse_args([])
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        for key, value in kwargs.items():
            hparams.__setattr__(key, value)
        module = DenseNet(
            inp_dim=hparams.inp_dim,
            hdims=hparams.hdims,
            outp_dim=hparams.outp_dim,
            linear_readout=hparams.linear_readout,
            nonlinearity=hparams.nonlinearity
        )
        super().__init__(module=module, scaling=hparams.scaling)
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--inp_dim', type=int, default=784*2)
        parser.add_argument('--hdims', type=int, nargs='*', default=[10000])
        parser.add_argument('--outp_dim', type=int, default=1)
        parser.add_argument('--linear_readout', action='store_true')
        parser.add_argument('--nonlinearity', choices=['piecewise_nonlinear', 'tanh'], default='piecewise_nonlinear')
        parser.add_argument('--scaling', type=float, default=1.)
        return parser

class DenseNet2(ModelWithNTK):
    def __init__(self, inp_dim, hdims=None, outp_dim=1, rho=1, bias=False, linear_readout=False, nonlinearity='piecewise_linear', scaling=1.):
        super().__init__()
        hdims = hdims or []
        L = []
        for i, (_in, _out) in enumerate(zip([inp_dim]+hdims[:-1], hdims)):
            linear = nn.Linear(_in, _out, bias=bias)
            if nonlinearity == 'piecewise_linear':
                nn.init.normal_(linear.weight, std=scaling*math.sqrt(2/(((B(rho)-1)**2)*_in)))
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
                nn.init.normal_(self.readout.weight, std=scaling*math.sqrt(2/(((B(rho)-1)**2)*_out)))
            elif nonlinearity == 'tanh':
                nn.init.xavier_normal_(self.readout.weight)
        self.linear_readout = linear_readout
    
    def parameters(self):
        if self.linear_readout:
            return self.readout.parameters()
        return super().parameters()
    
    def forward(self, x):
        x = self.features(x)
        x = self.readout(x)
        return x

class MTDenseNet2(ModelWithNTK):
    def __init__(self, inp_dim, hdims=None, outp_dim=1, rho=1, bias=False, linear_readout=False, nonlinearity='piecewise_linear', scaling=1., weight_scaling=None):
        super().__init__()
        hdims = hdims or []
        L = []
        if weight_scaling is None:
            weight_scaling = [1.]*outp_dim
        for i, (_in, _out) in enumerate(zip([inp_dim]+hdims[:-1], hdims)):
            linear = MTLinear(_in, _out, bias=bias, weight_scaling=weight_scaling)
            if nonlinearity == 'piecewise_linear':
                nn.init.normal_(linear.weight, std=scaling*math.sqrt(2/(((B(rho)-1)**2)*_in)))
            elif nonlinearity == 'tanh':
                nn.init.xavier_normal_(linear.weight)
            L.append(linear)
            if nonlinearity == 'piecewise_linear':
                L.append(ABReLU(rho))
            elif nonlinearity == 'tanh':
                L.append(nn.Tanh())
        self.features = MTSequential(*L)
        if len(hdims) > 0:
            _in = hdims[-1]
        else:
            _in = inp_dim
        self.readout = MTLinear(_in, outp_dim, bias=bias, weight_scaling=weight_scaling)
        if linear_readout:
            nn.init.zeros_(self.readout.weight)
        else:
            if nonlinearity == 'piecewise_linear':
                nn.init.normal_(self.readout.weight, std=scaling*math.sqrt(2/(((B(rho)-1)**2)*_out)))
            elif nonlinearity == 'tanh':
                nn.init.xavier_normal_(self.readout.weight)
        self.linear_readout = linear_readout
    
    def parameters(self):
        if self.linear_readout:
            return self.readout.parameters()
        return super().parameters()
    
    def forward(self, x, task):
        x = self.features(x, task)
        x = self.readout(x, task)
        return x
