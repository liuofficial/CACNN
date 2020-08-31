import numpy as np
import scipy.io as sio

def max_min(x):# The normalization in [-0.5, 0.5]
    return (x-np.min(x))/(np.max(x)-np.min(x))-0.5




