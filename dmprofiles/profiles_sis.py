import numpy as np

from astropy import units as u
from astropy.units import Quantity
from astropy.constants import G

######################
#### SIS Profiles ####
######################

def sigma_sis(r, vdisp):
    """
    SIS 2D density at r, given the velocity dispersion
    
    Inputs:
    r - radius at which to compute the mass density (with units)
    vdisp - velocity dispersion (with units)
    
    Returns:
    sigma_sis - 2D density at r (g / cm**2)
    """
    
    return (vdisp**2 / ( 2 * G * r )).to(u.g / u.cm**2)


def m_sis(r, vdisp):
    """
    SIS 2D cylindrical mass enclosed at r, given the velocity dispersion
    
    Inputs:
    r - radius at which to compute the enclosed mass (with units)
    vdisp - velocity despersion (with units)
    
    Returns:
    m_sis - 2D cylindrical integrated mass at r (Msun)
    """
    
    return ( ( np.pi * r * vdisp**2 ) / G ).to(u.Msun)