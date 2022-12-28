import numpy as np
import torch


# Parameters about learnable non-linear circuitry
eta = [-1., 0.1, 0.1, 0.1]
inv = [-1., 0.1, 0.1, 0.1]

# hardware-related parameters
gmin = 0.01
gmax = 10
m = 0.3
T = 0.1

# hyperparameter for pNN
topology=[11,7,5,3,4]

# variation-aware
N = 10    # repaet N times
epsilon = 0.1 # uniform distribution (1 - episilon, 1 + episilon)

