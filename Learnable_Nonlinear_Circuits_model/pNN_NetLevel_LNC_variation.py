import torch
import config
import copy
import numpy as np
#==========================================================================================================
#====================================== INV model =========================================================
#==========================================================================================================

Inv_model = torch.load(f'ApproximatorModel.nlc')
Inv_model.train(False)

#==========================================================================================================
#====================================== Learnable nonlieanr circuity ======================================
#==========================================================================================================

class Tanh(torch.nn.Module):
    def __init__(self, model, N, epsilon):
        super().__init__()
        self.N = N
        self.model= model
        self.eta_ = torch.nn.Parameter(torch.tensor(config.inv_rt_dr),requires_grad=True)
        self.eta_min = config.eta_min
        self.eta_max = config.eta_max
        self.inv_rt_min = config.inv_rt_min
        self.inv_rt_max = config.inv_rt_max
        self.inv_range_dr = config.inv_range_dr
        self.epsilon = epsilon
    @property
    def eta(self):
        self.eta_.data[0].clamp_(self.inv_rt_min[1], self.inv_rt_max[1])#r2
        self.eta_.data[1:5].clamp_(self.inv_rt_min[3:7], self.inv_rt_max[3:7])#r4, r5, w, l
        self.eta_.data[5].clamp_(self.inv_range_dr[0][0], self.inv_range_dr[0][1])#dr1
        self.eta_.data[6].clamp_(self.inv_range_dr[1][0], self.inv_range_dr[1][1])#dr2
        
        eta_rt = torch.zeros(10) 
        eta_rt[0] = (self.eta_[0] + self.eta_[5]).data.clamp_(self.inv_rt_min[0], self.inv_rt_max[0])
        eta_rt[1] = self.eta_[0]
        eta_rt[2] = (self.eta_[1] + self.eta_[6]).data.clamp_(self.inv_rt_min[2], self.inv_rt_max[2])
        eta_rt[3] = self.eta_[1]
        eta_rt[4] = self.eta_[2]
        eta_rt[5] = self.eta_[3]
        eta_rt[6] = self.eta_[4]
        
        return eta_rt
        
    
    @property
    def eta_extend(self):
        eta_mean = self.eta.repeat(self.N, 1)
        distribution = ((torch.rand(eta_mean.shape) * 2. - 1.) * self.epsilon) + 1.
        eta_variation = torch.mul(eta_mean, distribution)
        
        eta_extend = torch.zeros_like(eta_variation)
        eta_extend[:, :7] = eta_variation[:, :7]
        
        eta_extend[:,7] = eta_variation[:,1] / eta_variation[:, 0]
        eta_extend[:,8] = eta_variation[:,3] / eta_variation[:, 2]
        eta_extend[:,9] = eta_variation[:,6] / eta_variation[:, 5]
        
        # nomalization of r,t, ratio
        eta_rt_ = torch.mul((eta_extend-self.inv_rt_min),  1./(self.inv_rt_max - self.inv_rt_min))
        
        eta_temp = self.model(eta_rt_)
        eta_denom = torch.mul(eta_temp, (self.eta_max - self.eta_min)) + self.eta_min
        #print("eta", eta_denom)
        return eta_denom# (N,4)
    
    def forward(self, x):
        eta_extend = self.eta_extend
        x_temp = torch.zeros_like(x)
        for i in range(self.N):
            x_temp[i, :,:] = -eta_extend[i,0]  - eta_extend[i,1] * torch.tanh((x[i,:,:] - eta_extend[i,2]) * eta_extend[i,3])
        return  x_temp
    
    def SetParameter(self, name, value):
        if name == 'N':
            self.N = value
        elif name == 'epsilon':
            self.epsilon = value

class Inv(torch.nn.Module):
    def __init__(self, model, N, epsilon):
        super().__init__()
        
        self.N = N
        self.inv_ = torch.nn.Parameter(torch.tensor(config.inv_rt_dr), requires_grad=True)
        self.model= model
        self.eta_min = config.eta_min
        self.eta_max = config.eta_max
        self.inv_rt_min = config.inv_rt_min
        self.inv_rt_max = config.inv_rt_max
        self.inv_range_dr = config.inv_range_dr
        self.epsilon = epsilon
    
    
    @property
    def inv(self):
        self.inv_.data[0].clamp_(self.inv_rt_min[1], self.inv_rt_max[1])#r2
        self.inv_.data[1:5].clamp_(self.inv_rt_min[3:7], self.inv_rt_max[3:7])#r4, r5, w, l
        self.inv_.data[5].clamp_(self.inv_range_dr[0][0], self.inv_range_dr[0][1])#dr1
        self.inv_.data[6].clamp_(self.inv_range_dr[1][0], self.inv_range_dr[1][1])#dr2
        #  r,t, ratio  value
        inv_rt = torch.zeros(10)
        inv_rt[0] = (self.inv_[0] + self.inv_[5]).data.clamp_(self.inv_rt_min[0], self.inv_rt_max[0])
        inv_rt[1] = self.inv_[0]
        inv_rt[2] = (self.inv_[1] + self.inv_[6]).data.clamp_(self.inv_rt_min[2], self.inv_rt_max[2])
        inv_rt[3] = self.inv_[1]
        inv_rt[4] = self.inv_[2]
        inv_rt[5] = self.inv_[3]
        inv_rt[6] = self.inv_[4]

        return inv_rt
    
    @property
    def inv_extended(self):
        inv_mean = self.inv.repeat(self.N, 1)
        distribution = (torch.rand(inv_mean.shape) * 2. - 1.) * self.epsilon + 1.
        eta_variation = torch.mul(inv_mean, distribution)
        
        eta_extend = torch.zeros_like(eta_variation)
        eta_extend[:, :7] = eta_variation[:, :7]
        
        eta_extend[:,7] = eta_variation[:,1] / eta_variation[:, 0]
        eta_extend[:,8] = eta_variation[:,3] / eta_variation[:, 2]
        eta_extend[:,9] = eta_variation[:,6] / eta_variation[:, 5]
        
        # nomalization of r,t, ratio
        inv_rt_ = torch.mul((eta_extend-self.inv_rt_min),  1./(self.inv_rt_max - self.inv_rt_min))
        
        inv_temp = self.model(inv_rt_)
        inv_denom = torch.mul(inv_temp, self.eta_max - self.eta_min) + self.eta_min
        return inv_denom# (N,4)
    
    def forward(self, x):
        inv_extend = self.inv_extended
        x_temp = torch.zeros_like(x)
        for i in range(self.N):
            x_temp[i, :,:] = -(inv_extend[i,0] + inv_extend[i,1] * torch.tanh((x[i,:,:] - inv_extend[i,2]) * inv_extend[i,3]))
        return  x_temp
    
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
        
        theta = torch.rand(n_in + 2, n_out)/100 + config.gmin       
        theta[-1,:] = theta[-1,:] + config.gmax
        theta[-2,:] = (config.eta[2] / (1 - config.eta[2])) * (theta.sum(0) - theta[-2,:])
        
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)
        self.Inv = Inv
        self.N = N
        self.epsilon = epsilon
    @property
    def theta(self):
        self.theta_.data.clamp_(-config.gmax, config.gmax)
        theta_temp = self.theta_.clone()
        theta_temp[theta_temp.abs() < config.gmin] = 0. 
        return theta_temp.detach() - self.theta_.detach() + self.theta_
    
    @property
    def w(self):
        theta_mean = self.theta.repeat(self.N, 1, 1)
        distribution = ((torch.rand(theta_mean.shape) * 2. - 1.) * self.epsilon) + 1.
        theta_variation = torch.mul(theta_mean, distribution)
        return theta_variation.abs()/(theta_variation.abs().sum(dim=1, keepdim=True))# (N,In,Out)
       
    def forward(self, a): 
        vb = torch.ones((a.shape[0], a.shape[1], 1))
        vd = torch.zeros((a.shape[0], a.shape[1], 1))
        a_extended = torch.cat((a, vb, vd), 2)
       
        a_neg = self.Inv(a_extended)
        a_neg[:,:,-1] = 0.
        pt = torch.sign(self.theta)
        pt[pt<0] = 0.
        nt = torch.ones_like(pt) - pt
        #print(self.w.shape, pt.shape)
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
        self.epsilon = epsilon
        self.activation = Tanh(Inv_model, self.N, self.epsilon)
        self.Inv = Inv(Inv_model, self.N, self.epsilon)
        
        self.model = torch.nn.Sequential()
        for l in range(len(topology)-1):
            self.model.add_module(f'pLayer {l}', pLayer(topology[l], topology[l+1], self.Inv, self.N, self.epsilon))
            self.model.add_module(f'pTanh {l}' , self.activation)
            
    def forward(self, x):
        x_extend = x.repeat(self.N, 1, 1)
        return self.model(x_extend)
    
    def GetParam(self, name):
        if name=='eta_':
            return [p for name,p in self.model.named_parameters() if name.endswith('.eta_')]
        elif name=='theta_':
            return [p for name,p in self.model.named_parameters() if name.endswith('.theta_')]
        elif name=='inv_':
            return [p for name,p in self.model.named_parameters() if name.endswith('.inv_')]

    def SetParameter(self, name, value):
        if name == 'N':
            self.N = value
            for l in self.model:
                l.SetParameter(name, value)
        elif name == 'epsilon':
            for l in self.model:
                l.SetParameter(name, value)
        
def lossfunction(prediction,label):   
    N = prediction.shape[0]
    label = label.reshape(-1, 1).repeat(N, 1, 1) #[N, E, 1]
    fy = prediction.gather(2, label)[:, :, 0] #[N, E]
    fny = prediction.clone()
    fny = fny.scatter_(2, label, -10 ** 10)
    fnym = torch.max(fny, axis=2).values #[N, E]
    l = torch.max(config.m + config.T - fy, torch.tensor(0)) + torch.max(config.m + fnym, torch.tensor(0))
    L = torch.mean(l)
    return L