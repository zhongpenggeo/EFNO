'''
deeponet: proposed by Lu lu
    - branch net: FNO 
    - trunck net: FNN

FNO2d module is cloned from Zongyi Li's code, https://github.com/zongyi-li/fourier_neural_operator
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

# activattion type
act_dict = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "softplus": nn.Softplus(),
    "sigmoid": nn.Sigmoid(),
    "idt": nn.Identity(),
    "gelu": nn.GELU()
}

# initiation method
init_dict={
    "xavier_normal": nn.init.xavier_normal_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "uniform": nn.init.uniform_,
    "norm": nn.init.normal_
}

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.   

        Parameters:
        -----------
        in_channels  : lifted dimension 
        out_channels : output dimension 
        modes1       : truncated modes in the first dimension of fourier domain 
        modes2       : truncated modes in the second dimension of fourier domain
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        # the result in last dimension is half for fft
        # and the result in  the second to last dimension is symmetric.
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2) # because of  symmetry?

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

# branch net
class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width,n_out,layer_num=4,last_size=128,act_func="gelu"):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        Parameters:
        -----------
            - modes1    : truncated modes in the first dimension of fourier domain
            - modes2    : truncated modes in the second dimension of fourier domain
            - width     : width of the lifted dimension
            - n_out     : output dimension, here is 4: rhoxy, phasexy, rhoyx, phaseyx
            - layer_num : number of fourier layers
            - last_size : width of projected dimension
            - act_func  : activation function, key must in act_dict
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width # lift to width dimensions
        self.padding = 9 # pad the domain if input is non-periodic
        self.layer_num = layer_num
        if act_func in act_dict.keys():
            self.activation = act_dict[act_func]
        else:
            raise KeyError("act name not in act_dict")
        self.fc0 = nn.Linear(1, self.width) # input channel is 3: (a(x, y), x, y)

        self.fno = nn.ModuleList()
        self.conv = nn.ModuleList()
        for _ in range(layer_num):
            self.fno.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.conv.append(nn.Conv2d(self.width, self.width, 1))

        self.fc1 = nn.Linear(self.width, last_size)
        # notice!
        self.fc2 = nn.Linear(last_size, n_out)

    def forward(self, x):
        '''
        input  : (batch, x, y, 1)
        output : (batch, x, y, n_out)
        '''
        # lift to high dimension
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)# batch_size, width, n1,n2
        x = F.pad(x, [0,self.padding, 0,self.padding])# pad for last 2 dimensions (n1,n2)
        # number of fourier layers
        for i in range(self.layer_num):
            x1 = self.fno[i](x)
            x2 = self.conv[i](x)
            x  = x1 + x2
            x = self.activation(x)

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        # 
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        
        return x