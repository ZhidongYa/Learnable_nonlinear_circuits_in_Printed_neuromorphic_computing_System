import torch
import copy
import config
import numpy as np
import pickle

#==========================================================================================================
#====================================== Learnable nonlieanr circuity ======================================
#==========================================================================================================

class TanhRT(torch.nn.Module):
    def __init__(self, N, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.N = N
        # R1n, k1, R3n, k2, R5n, Wn, Ln
        # note: k1, k2 are just values to keep R2<R1 and R4<R3
        self.rt_ = torch.nn.Parameter(torch.tensor(config.rt_), requires_grad=True)
        # model
        self.eta_estimator = torch.load('./NN/model.nlc')
        self.eta_estimator.train(False)
        
        with open('./dataset/inv_dataset.p', 'rb') as f:
            data = pickle.load(f)
        self.X_max = data['X_max']
        self.X_min = data['X_min']
        self.Y_max = data['Y_max']
        self.Y_min = data['Y_min']
    
    @property
    def RT_(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_)
        # calculate normalized (only R1n, R3n, R5n, Wn, Ln)
        RTn = torch.zeros([10])
        RTn[0] = rt_temp[0]    # R1n
        RTn[2] = rt_temp[2]    # R3n
        RTn[4] = rt_temp[4]    # R5n
        RTn[5] = rt_temp[5]    # Wn
        RTn[6] = rt_temp[6]    # Ln
        # denormalization
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        # calculate R2, R4
        R2 = RT[0] * rt_temp[1] # R2 = R1 * k1
        R4 = RT[2] * rt_temp[3] # R4 = R3 * k2
        # stack new variable: R1, R2, R3, R4, R5, W, L
        RT_full = torch.stack([RT[0], R2, RT[2], R4, RT[4], RT[5], RT[6]])
        return RT_full
    
    @property
    def RT(self):
        RT_full = torch.zeros([10])
        RT_full[:7] = self.RT_.clone()
        RT_full[RT_full>self.X_max] = self.X_max[RT_full>self.X_max]
        RT_full[RT_full<self.X_min] = self.X_min[RT_full<self.X_min]
        return RT_full[:7].detach() + self.RT_ - self.RT_.detach()
    
    @property
    def RT_full(self):
        R1 = self.RT[0]
        R2 = self.RT[1]
        R3 = self.RT[2]
        R4 = self.RT[3]
        R5 = self.RT[4]
        W  = self.RT[5]
        L  = self.RT[6]
        k1 = R2 / R1
        k2 = R4 / R3
        k3 = L / W
        return torch.stack([R1, R2, R3, R4, R5, W, L, k1, k2, k3])
    
    @property
    def RTn_full(self):
        return (self.RT_full - self.X_min) / (self.X_max - self.X_min) 
        
    @property
    def RT_extend(self):
        RT_mean = self.RTn_full.repeat(self.N, 1)
        variance = ((torch.rand(RT_mean.shape) * 2. - 1.) * self.epsilon) + 1.
        RT_variation = torch.mul(RT_mean, variance)
        return RT_variation
    
    @property
    def eta_extend(self):
        eta_n = self.eta_estimator(self.RT_extend)
        eta = eta_n * (self.Y_max - self.Y_min) + self.Y_min
        return eta
    
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

class InvRT(torch.nn.Module):
    def __init__(self, N, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.N = N
        # R1n, k1, R3n, k2, R5n, Wn, k3
        # be careful, k1, k2, k3 are not normalized
        self.rt_ = torch.nn.Parameter(torch.tensor(config.rt_), requires_grad=True)
        # model
        self.eta_estimator = torch.load('./NN/model.nlc')
        self.eta_estimator.train(False)
        
        with open('./dataset/inv_dataset.p', 'rb') as f:
            data = pickle.load(f)
        self.X_max = data['X_max']
        self.X_min = data['X_min']
        self.Y_max = data['Y_max']
        self.Y_min = data['Y_min']
    
    @property
    def RT_(self):
        # keep values in (0,1)
        rt_temp = torch.sigmoid(self.rt_)
        # calculate normalized (only R1n, R3n, R5n, Wn, Ln)
        RTn = torch.zeros([10])
        RTn[0] = rt_temp[0]    # R1n
        RTn[2] = rt_temp[2]    # R3n
        RTn[4] = rt_temp[4]    # R5n
        RTn[5] = rt_temp[5]    # Wn
        RTn[6] = rt_temp[6]    # Ln
        # denormalization
        RT = RTn * (self.X_max - self.X_min) + self.X_min
        # calculate R2, R4
        R2 = RT[0] * rt_temp[1] # R2 = R1 * k1
        R4 = RT[2] * rt_temp[3] # R4 = R3 * k2
        # stack new variable: R1, R2, R3, R4, R5, W, L
        RT_full = torch.stack([RT[0], R2, RT[2], R4, RT[4], RT[5], RT[6]])
        return RT_full
    
    @property
    def RT(self):
        RT_full = torch.zeros([10])
        RT_full[:7] = self.RT_.clone()
        RT_full[RT_full>self.X_max] = self.X_max[RT_full>self.X_max]
        RT_full[RT_full<self.X_min] = self.X_min[RT_full<self.X_min]
        return RT_full[:7].detach() + self.RT_ - self.RT_.detach()
    
    @property
    def RT_full(self):
        R1 = self.RT[0]
        R2 = self.RT[1]
        R3 = self.RT[2]
        R4 = self.RT[3]
        R5 = self.RT[4]
        W  = self.RT[5]
        L  = self.RT[6]
        k1 = R2 / R1
        k2 = R4 / R3
        k3 = L / W
        return torch.stack([R1, R2, R3, R4, R5, W, L, k1, k2, k3])
    
    @property
    def RTn_full(self):
        return (self.RT_full - self.X_min) / (self.X_max - self.X_min) 
    
    @property
    def RT_extend(self):
        RT_mean = self.RTn_full.repeat(self.N, 1)
        variance = ((torch.rand(RT_mean.shape) * 2. - 1.) * self.epsilon) + 1.
        RT_variation = torch.mul(RT_mean, variance)
        return RT_variation
    
    @property
    def eta_extend(self):
        eta_n = self.eta_estimator(self.RT_extend)
        eta = eta_n * (self.Y_max - self.Y_min) + self.Y_min
        return eta
    
    def forward(self, z):
        eta_extend = self.eta_extend
        a = torch.zeros_like(z)
        for i in range(self.N):
            a[i,:,:] = - (eta_extend[i,0] + eta_extend[i,1] * torch.tanh((z[i,:,:] - eta_extend[i,2]) * eta_extend[i,3]))
        return a
    
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
        self.activation = TanhRT(N, epsilon)
        self.Inv = InvRT(N, epsilon)
        
        self.model = torch.nn.Sequential()
        for l in range(len(topology)-1):
            self.model.add_module(f'pLayer {l}', pLayer(topology[l], topology[l+1], self.Inv, N, epsilon))
            self.model.add_module(f'pTanh {l}' , self.activation)
            
    def forward(self, x):
        x_extend = x.repeat(self.N, 1, 1)
        return self.model(x_extend)
    
    def GetParam(self, name):
        if name=='rt':
            return [p for name,p in self.named_parameters() if name.endswith('.rt_')]
        elif name=='theta':
            return [p for name,p in self.named_parameters() if name.endswith('.theta_')]

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