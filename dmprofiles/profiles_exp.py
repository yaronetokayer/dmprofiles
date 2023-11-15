import numpy as np

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.units import Quantity
from astropy.constants import G

from scipy.special import kn

# Define a default cosmology

cosmo = FlatLambdaCDM(H0=69, Om0=0.3)

##############################
#### Exponential Profiles ####
##############################

def rho_exp_3d(r, m_0, r_d):
    '''
    3D mass enclosed within radius r for a 3D exponential disk
    
    Inputs:
    r - aperture radius (Astropy distance units)
    m_0 - total mass in the exponential disk,
          normalization of the 3D *mass profile* (Astropy mass units)
          The normalization of the 3D density is 1/(4*pi*r_d**2) that of the 3D mass profile
    r_d - scale radius of the 3D exponential profile (Astropy distance units)
    
    Returns:
    m_exp_3d - mass enclosed (Msun)
    '''

    rho_3d = ( m_0 / ( 4 * np.pi * r_d**2 * r ) ) * np.exp(-r/r_d)

    return ( rho_3d ).to(u.Msun / u.kpc**3)

def m_exp_3d(r, m_0, r_d):
    '''
    3D mass enclosed within radius r for a 3D exponential disk
    
    Inputs:
    r - aperture radius (Astropy distance units)
    m_0 - total mass in the exponential disk,
          normalization of the 3D mass profile - total (Astropy units)
    r_d - scale radius of the 3D exponential disk (Astropy distance units)
    
    Returns:
    m_exp_3d - mass enclosed (Msun)
    '''
    
    m_exp_3d = m_0 * ( 1 - ( 1 + r / r_d ) * np.exp(-r/r_d) )
    
    return m_exp_3d.to(u.Msun)

def sigma_exp_3d(r, m_0, r_d):
    '''
    The 2D projected surface mass density for the 3D exponential disk
    NOTE: This is not itself an exponential function
    It the zeroth order modified Bessel function of the second kind

    r - aperture radius (Astropy distance units)
    m_0 - total mass in the exponential disk,
          normalization of the 3D mass profile - total (Astropy units)
    r_d - scale radius of the 3D exponential disk (Astropy distance units)
    
    Returns:
    sigma_exp_3d - 2D projected surface mass density (Msun/kpc**2)
    '''

    x = ( r / r_d ).to(u.dimensionless_unscaled).value

    sigma_exp_3d = m_0 / ( 2 * np.pi * r_d**2 ) * kn(0, x)

    return sigma_exp_3d.to(u.Msun / u.kpc**2)

def m_exp_3d_2d(r, m_0, r_d):
    '''
    The 2D projected enclosed mass for the 3D exponential disk
    NOTE: This is not itself an exponential function
    It uses a first order modified Bessel function of the second kind

    r - aperture radius (Astropy distance units)
    m_0 - total mass in the exponential disk,
          normalization of the 3D mass profile (Astropy units)
    r_d - scale radius of the 3D exponential disk (Astropy distance units)
    
    Returns:
    m_exp_3d_2d - 2D projected surface mass density (Msun/kpc**2)
    '''  

    x = ( r / r_d ).to(u.dimensionless_unscaled).value

    m_exp_3d_2d = m_0 * (1 - x * kn(1, x))

    return m_exp_3d_2d.to(u.Msun)
    
def m_exp_2d_2d(r, m_0, r_d):
    '''
    2D mass enclosed within radius r for a 2D exponential disk
    
    Inputs:
    r - aperture radius (Astropy distance units)
    m_0 - total mass in the exponential disk,
          normalization of the 2D mass profile (Astropy units)
    r_d - scale radius of the 2D exponential disk (Astropy distance units)
    
    Returns:
    m_exp_2d - mass enclosed (Msun)
    '''

    m_exp_2d = m_0 * ( 1 - ( 1 + r / r_d ) * np.exp(-r/r_d) )
    
    return m_exp_2d.to(u.Msun)

def m_exp_2d_tot(sig_0, r_d):
    '''
    Total mass for a 2D exponential disk
    
    Inputs:
    sig_0 - normalization of the 2D exponential disk (Astropy units)
    r_d - scale radius of the 2D exponential disk (Astropy distance units)
    
    Returns:
    m_exp_tot - total integrated mass (Msun)
    '''
    
    m_exp_tot = 2 * np.pi * sig_0 * r_d**2
    
    return m_exp_tot.to(u.Msun)

def m_0_from_m_exp_2d_2d(m_enc, r, r_d):
    '''
    Get the exponential disk normalization, given the mass enclosed within a 
    2D exponential disk aperture
    
    Inputs:
    m_enc - known 2D projected mass within r (Astropy units)
    r - radius within which mass is known (Astropy distance units)
    r_d - scale radius of the 2D exponential disk (Astropy distance units)
    
    Returns:
    m_0 - 2D exponential disk normalization (Astropy units)
    '''
    
    m_0 = m_enc / ( 1 - ( 1 + r / r_d ) * np.exp(-r/r_d) )
    
    return m_0.to(u.Msun)

def m_0_from_m_exp_3d_2d(m_enc, r, r_d):
    '''
    Get the exponential disk normalization, given the mass enclosed within a 
    3D exponential disk aperture
    
    Inputs:
    m_enc - known 2D projected mass within r (Astropy units)
    r - radius within which mass is known (Astropy distance units)
    r_d - scale radius of the 2D exponential disk (Astropy distance units)
    
    Returns:
    m_0 - 3D exponential disk normalization (Astropy units)
    '''
    x = ( r / r_d ).to(u.dimensionless_unscaled).value
    
    m_0 = m_enc / (1 - x * kn(1, x))
    
    return m_0.to(u.Msun)