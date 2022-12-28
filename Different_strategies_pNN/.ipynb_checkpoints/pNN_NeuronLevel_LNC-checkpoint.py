import torch
import config
import copy

#==========================================================================================================
#============================================ printed Neuron ==============================================
#==========================================================================================================

class pNeuron(torch.nn.Module):
    def __init__(self, n_in):
        super().__init__()
        
        theta = torch.rand(n_in + 2, 1) / 100 + config.gmin
        theta[-1,:] = theta[-1,:] + config.gmax
        theta[-2,:] = (config.eta[2] / (1 - config.eta[2])) * (theta.sum(0) - theta[-2,:])
        
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)  
        self.eta_ = torch.nn.Parameter(torch.tensor(config.eta), requires_grad=True)
        self.inv_ = torch.nn.Parameter(torch.tensor(config.inv), requires_grad=True)
        
    @property
    def theta(self):
        self.theta_.data.clamp_(-config.gmax, config.gmax)
        theta_temp = self.theta_.clone()
        theta_temp[theta_temp.abs() < config.gmin] = 0. 
        return theta_temp.detach() - self.theta_.detach() + self.theta_
    
    @property
    def eta(self):
        eta_temp = torch.zeros_like(self.eta_)    
        eta_temp[0] = self.eta_[0]
        eta_temp[1] = torch.nn.functional.softplus(self.eta_[1])
        eta_temp[2] = self.eta_[2]
        eta_temp[3] = torch.nn.functional.softplus(self.eta_[3])    
        return eta_temp  
    
    @property
    def inv(self):
        inv_temp = torch.zeros_like(self.inv_)    
        inv_temp[0] = self.inv_[0]
        inv_temp[1] = torch.nn.functional.softplus(self.inv_[1])
        inv_temp[2] = self.inv_[2]
        inv_temp[3] = torch.nn.functional.softplus(self.inv_[3])    
        return inv_temp
    
    @property
    def w(self):
        return self.theta.abs()/(self.theta.abs().sum(dim=0))
    
    def Tanh(self,x):
        return self.eta[0] + self.eta[1] * torch.tanh((x - self.eta[2]) * self.eta[3])
    
    def Inv(self,x):
        return -(self.inv[0] + self.inv[1] * torch.tanh((x - self.inv[2]) * self.inv[3]))
    
    def linear(self, a): 
        vb = torch.ones((a.shape[0], 1))
        vd = torch.zeros((a.shape[0], 1))
        a_extended = torch.cat((a, vb, vd), 1)
        
        a_neg = self.Inv(a_extended)
        a_neg[:,-1] = 0.
        
        pt = torch.sign(self.theta)
        pt[pt<0] = 0.
        nt = torch.ones_like(pt) - pt

        return torch.matmul(a_extended, torch.mul(self.w, pt)) + torch.matmul(a_neg, torch.mul(self.w, nt))    
    
    def forward(self, x):         
        return self.Tanh(self.linear(x))


#==========================================================================================================
#============================================== printed Layer =============================================
#==========================================================================================================

class pLayer(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.neurons = torch.nn.ModuleList([pNeuron(n_in) for i in range(n_out)])
        
    def forward(self,x):
        result = [n.forward(x) for n in self.neurons]        
        return torch.cat(result, dim=1)

    
#==========================================================================================================
#============================================== printed NN ================================================
#==========================================================================================================

class pNN(torch.nn.Module):
    def __init__(self, topology):
        super().__init__()
        self.model = torch.nn.Sequential()
        for l in range(len(topology)-1):
            self.model.add_module(f'pLayer {l}', pLayer(topology[l], topology[l+1]))
        
    def forward(self, x):           
        return self.model(x)
                                       
    def GetParam(self, name):
        if name=='eta':
            return [p for name,p in self.model.named_parameters() if name.endswith('.eta_')]
        elif name=='theta':
            return [p for name,p in self.model.named_parameters() if name.endswith('.theta_')]
        elif name=='inv':
            return [p for name,p in self.model.named_parameters() if name.endswith('.inv_')]

#==========================================================================================================
#============================================ Loss function ===============================================
#==========================================================================================================

def lossfunction(prediction,label):   
    label = label.reshape(-1, 1)
    fy = prediction.gather(1, label).reshape(-1, 1)
    fny = prediction.clone()
    fny = fny.scatter_(1, label, -10 ** 10)
    fnym = torch.max(fny, axis=1).values.reshape(-1, 1)
    l = torch.max(config.m + config.T - fy, torch.tensor(0)) + torch.max(config.m + fnym, torch.tensor(0))
    L = torch.mean(l)
    return L