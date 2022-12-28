import torch
import copy
import config
import numpy as np

#==========================================================================================================
#====================================== Learnable nonlieanr circuity ======================================
#==========================================================================================================

class Tanh(torch.nn.Module):
    def __init__(self, N, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.N = N
        self.eta_ = torch.nn.Parameter(torch.tensor(config.eta),requires_grad=True)
        
    @property
    def eta(self):
        eta_temp = torch.zeros_like(self.eta_)    
        eta_temp[0] = self.eta_[0]
        eta_temp[1] = torch.nn.functional.softplus(self.eta_[1])
        eta_temp[2] = self.eta_[2]
        eta_temp[3] = torch.nn.functional.softplus(self.eta_[3])    
        return eta_temp
    
    @property
    def eta_extend(self):
        eta_mean = self.eta.repeat(self.N, 1)
        variance = ((torch.rand(eta_mean.shape) * 2. - 1.) * self.epsilon) + 1.
        eta_variation = torch.mul(eta_mean, variance)
        return eta_variation
    
    def forward(self, z):
        eta_extend = self.eta_extend
        a = torch.zeros_like(z)
        for i in range(self.N):
            a[i,:,:] = eta_extend[i,0] + eta_extend[i,1] * torch.tanh((z[i,:,:] - eta_extend[i,2]) * eta_extend[i,3])
        return a
    
    def SetParameter(self, name, value):
        if name == 'N':
            self.N = value
        elif name == 'epsilon':
            self.epsilon = value

class Inv(torch.nn.Module):
    def __init__(self, N, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.N = N
        self.inv_ = torch.nn.Parameter(torch.tensor(config.inv), requires_grad=True)
        
    @property
    def inv(self):
        inv_temp = torch.zeros_like(self.inv_)    
        inv_temp[0] = self.inv_[0]
        inv_temp[1] = torch.nn.functional.softplus(self.inv_[1])
        inv_temp[2] = self.inv_[2]
        inv_temp[3] = torch.nn.functional.softplus(self.inv_[3])    
        return inv_temp
    
    @property
    def inv_extend(self):
        inv_mean = self.inv.repeat(self.N, 1)
        variance = ((torch.rand(inv_mean.shape) * 2. - 1.) * self.epsilon) + 1.
        inv_variation = torch.mul(inv_mean, variance)
        return inv_variation
    
    def forward(self, x):
        inv_extend = self.inv_extend
        inv_x = torch.zeros_like(x)
        for i in range(self.N):
            inv_x[i,:,:] = - (inv_extend[i,0] + inv_extend[i,1] * torch.tanh((x[i,:,:] - inv_extend[i,2]) * inv_extend[i,3]))
        return inv_x
    
    def SetParameter(self, name, value):
        if name == 'N':
            self.N = value
        elif name == 'epsilon':
            self.epsilon = value
    
#==========================================================================================================
#============================================= printed Layer ==============================================
#==========================================================================================================

class pLayer(torch.nn.Module):
    def __init__(self, n_in, n_out, Inv, N, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.N = N
        theta = torch.rand(n_in + 2, n_out)/100 + config.gmin       
        theta[-1,:] = theta[-1,:] + config.gmax
        theta[-2,:] = (config.eta[2] / (1 - config.eta[2])) * (theta.sum(0) - theta[-2,:])
        
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)
        self.Inv = Inv
        
    @property
    def theta(self):
        self.theta_.data.clamp_(-config.gmax, config.gmax)
        theta_temp = self.theta_.clone()
        theta_temp[theta_temp.abs() < config.gmin] = 0. 
        return theta_temp.detach() - self.theta_.detach() + self.theta_
    
    @property
    def w(self):
        theta_mean = self.theta.repeat(self.N, 1, 1)
        variance = ((torch.rand(theta_mean.shape) * 2. - 1.) * self.epsilon) + 1.
        theta_variation = torch.mul(theta_mean, variance)
        return theta_variation.abs()/(theta_variation.abs().sum(dim=1, keepdim=True))
       
    def forward(self, a): 
        vb = torch.ones((self.N, a.shape[1], 1))
        vd = torch.zeros((self.N, a.shape[1], 1))
        a_extended = torch.cat([a, vb, vd], dim=2)
       
        a_neg = self.Inv(a_extended)
        a_neg[:,:,-1] = 0.
        
        pt = torch.sign(self.w)
        pt[pt<0] = 0.
        nt = torch.ones_like(pt) - pt
        
        return torch.matmul(a_extended, torch.mul(self.w, pt)) + torch.matmul(a_neg, torch.mul(self.w, nt))
    
    def SetParameter(self, name, value):
        if name == 'N':
            self.N = value
            self.Inv.SetParameter(name, value)
        elif name == 'epsilon':
            self.epsilon = value
            self.Inv.SetParameter(name, value)
            
#==========================================================================================================
#============================================== printed NN ================================================
#==========================================================================================================
    
class pNN(torch.nn.Module):
    def __init__(self, topology, N, epsilon):
        super().__init__()
        self.N = N
        self.activation = Tanh(N, epsilon)
        self.Inv = Inv(N, epsilon)
        
        self.model = torch.nn.Sequential()
        for l in range(len(topology)-1):
            self.model.add_module(f'pLayer {l}', pLayer(topology[l], topology[l+1], self.Inv, N, epsilon))
            self.model.add_module(f'pTanh {l}' , self.activation)
            
    def forward(self, x):
        x_extend = x.repeat(self.N, 1, 1)
        return self.model(x_extend)
    
    def GetParam(self, name):
        if name=='eta':
            return [p for name,p in self.named_parameters() if name.endswith('.eta_')]
        elif name=='theta':
            return [p for name,p in self.named_parameters() if name.endswith('.theta_')]
        elif name=='inv':
            return [p for name,p in self.named_parameters() if name.endswith('.inv_')]

    def SetParameter(self, name, value):
        if name == 'N':
            self.N = value
            for l in self.model:
                l.SetParameter(name, value)
        elif name == 'epsilon':
            for l in self.model:
                l.SetParameter(name, value)

#==========================================================================================================
#============================================ Loss function ===============================================
#==========================================================================================================

def lossfunction(prediction,label):
    N = prediction.shape[0]
    label = label.reshape(-1, 1).repeat(N, 1, 1) #[N, E, N_class]
    fy = prediction.gather(2, label)[:, :, 0] #[N, E]
    fny = prediction.clone()
    fny = fny.scatter_(2, label, -10 ** 10)
    fnym = torch.max(fny, axis=2).values #[N, E]
    l = torch.max(config.m + config.T - fy, torch.tensor(0)) + torch.max(config.m + fnym, torch.tensor(0))
    L = torch.mean(l)
    return L       