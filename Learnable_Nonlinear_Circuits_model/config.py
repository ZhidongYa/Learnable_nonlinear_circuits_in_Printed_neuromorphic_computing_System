import numpy as np
import torch
import torch


# Parameters about learnable non-linear circuitry
#eta = [ 4.9564e-02,  1.0491e+00, -6.6612e-03,  1.1778e+01]
eta=[4.5771e-03, 1.0046e+00, 8.4388e-04, 4.2005e+00]

# inv

inv_rt_dr1= [100, 200000, 200000, 4.0000e-04, 5.0000e-05, 100, 100000] # r2,4, 5, w, l, dr1, dr2
inv_rt_dr= [178.0,  132000.0, 23000.0, 0.00023, 2.4e-05, 266., 83000]
inv_rt_min = torch.tensor([1.0000e+01, 5.0000e+00, 1.0000e+04, 8.0000e+03, 1.0000e+04, 2.0000e-04, 1.0000e-05, 1.0352e-02, 2.3454e-02, 1.2579e-02])

inv_rt_max = torch.tensor([5.0000e+02, 2.5000e+02, 5.0000e+05, 4.0000e+05, 5.0000e+05, 8.0000e-04, 7.0000e-05, 9.8000e-01, 9.9749e-01, 3.4328e-01])
inv_range_dr = [ [0, 495], [0, 492000]  ]#the range of dr1, dr2

eta_min = torch.tensor([-0.3552,  0.6549, -1.9425,  0.8018])
eta_max = torch.tensor([ 0.6198,  1.5777,  1.6535, 76.5687])

# hardware-related parameters
gmin = 0.01
gmax = 10
m = 0.3
T = 0.1

# hyperparameter for pNN
#topology=[4,3,3]

# variation-aware
N = 10    # repaet N times
test = 100
#epsilon = 0.05 # uniform distribution (1 - episilon, 1 + episilon)

