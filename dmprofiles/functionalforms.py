'''
Functional forms for fitting
'''

import numpy as np
from sigfig import round

###########################
#### Powerlaw function ####
###########################

def powerlaw(x, k, alpha):
    """
    Powerlaw at x
    
    Inputs:
    x - array-like, values at which to compute powerlaw output
    k - scalar, normalization
    alpha - scalar, slope of the powerlaw
    
    Returns:
    Powerlaw calculated at x
    """
    
    return k * (x**alpha)