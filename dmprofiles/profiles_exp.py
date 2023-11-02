import numpy as np

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.units import Quantity
from astropy.constants import G


# Define a default cosmology

cosmo = FlatLambdaCDM(H0=69, Om0=0.3)

##############################
#### Exponential Profiles ####
##############################

def rho_exp(r, m_0, r_d):
    '''
    3D mass enclosed within radius r for an exponential disk
    
    Inputs:
    r - aperture radius (Astropy distance units)
    m_0 - normalization of the 3D *mass profile* (Astropy mass units)
          The normalization of the 3D density is 1/(4*pi*r_d**2) that of the 3D mass profile
    r_d - scale radius of the 3D exponential profile (Astropy distance units)
    
    Returns:
    m_exp_3d - mass enclosed (Msun)
    '''

    rho_3d = ( m_0 / ( 4 * np.pi * r_d**2 * r ) ) * np.exp(-r/r_d)

    return ( rho_3d ).to(u.Msun / u.kpc**3)

def m_exp_3d(r, m_0, r_d):
    '''
    3D mass enclosed within radius r for an exponential disk
    
    Inputs:
    r - aperture radius (Astropy distance units)
    m_0 - normalization of the 3D mass profile (Astropy units)
    r_d - scale radius of the 3D exponential disk (Astropy distance units)
    
    Returns:
    m_exp_3d - mass enclosed (Msun)
    '''
    
    m_exp_3d = m_0 * ( 1 - ( 1 + r / r_d ) * np.exp(-r/r_d) )
    
    return m_exp_3d.to(u.Msun)

def m_exp_2d(r, sig_0, r_d):
    '''
    THIS IS FISHY!! - DO NOT TRUST
    2D mass enclosed within radius r for an exponential disk
    
    Inputs:
    r - aperture radius (Astropy distance units)
    sig_0 - normalization of the 2D exponential disk (Astropy units)
    r_d - scale radius of the 2D exponential disk (Astropy distance units)
    
    Returns:
    m_exp_2d - mass enclosed (Msun)
    '''
    
    m_exp_2d = 2 * np.pi * sig_0 * r_d**2 * ( 1 - ( 1 + r / r_d ) * np.exp(-r/r_d) )
    
    return m_exp_2d.to(u.Msun)

def m_exp_tot(sig_0, r_d):
    '''
    Total mass for an exponential disk
    
    Inputs:
    sig_0 - normalization of the 2D exponential disk (Astropy units)
    r_d - scale radius of the 2D exponential disk (Astropy distance units)
    
    Returns:
    m_exp_tot - total integrated mass (Msun)
    '''
    
    m_exp_tot = 2 * np.pi * sig_0 * r_d**2
    
    return m_exp_tot.to(u.Msun)

def sig_0_from_m_exp_2d(m_enc, r, r_d):
    '''
    Get the exponential disk normalization, given the mass enclosed within a 
    2D exponential disk aperture
    
    Inputs:
    m_enc - known 2D projected mass within r (Astropy units)
    r - radius within which mass is known (Astropy distance units)
    r_d - scale radius of the 2D exponential disk (Astropy distance units)
    
    Returns:
    sig_0 - 2D exponential disk normalization (Astropy units)
    '''
    
    sig_0 = m_enc / ( 2 * np.pi * r_d**2 * ( 1 - ( 1 + r / r_d ) * np.exp(-r/r_d) ) )
    
    return sig_0.to(u.Msun / u.kpc**2)