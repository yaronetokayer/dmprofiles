import numpy as np

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.units import Quantity
from astropy.constants import G


# Define a default cosmology

cosmo = FlatLambdaCDM(H0=69, Om0=0.3)

########################
#### PIEMD Profiles ####
########################

def m_piemd_2d(r, vdisp, rcore, rcut, convention='eliasdottir+07'):
    """
    PIEMD 2D cylindrical mass enclosed at r, given the velocity dispersion, r_core and r_cut
    
    Inputs:
    r - radius at which to compute the enclosed mass (Astropy distance units expected)
    vdisp - velocity dispersion of galaxy to set the norm of the mass distribution model  (Astropy units expected)
    rcore - PIEMD core radius  of galaxy (Astropy distance units expected)
    rcut - PIEMD cut radius  of galaxy (Astropy distance units expected)
    convention - which convention of dispersion velocity is being used. Options are 'eliasdottir+07' and 'sis'
    
    Returns:
    m_piemd_2d - 2D cylindrical integrated mass at r (Msun)
    """
    
    if convention=='eliasdottir+07':
        sigma_0 = vdisp**2 / ( 4 / 3 * G ) * (rcut**2 - rcore**2) / (rcore * rcut**2)
    elif convention=='sis':
        sigma_0 = ((vdisp)**2) / ( 2 * G * rcore)
    else:
        raise ValueError("convention keyword can be 'eliasdottir+07' or 'sis'")
    
    m_piemd_2d = ( ( 2 * np.pi * sigma_0 * rcore * rcut / (rcut - rcore) ) 
                  * ( np.sqrt(rcore**2 + r**2) 
                     - np.sqrt(rcut**2 + r**2) 
                     + rcut - rcore ) )
    
    return m_piemd_2d.to(u.Msun)

def m_piemd_tot(vdisp, rcore, rcut):
    """
    PIEMD total mass enclosed, given the velocity dispersion, r_core and r_cut
    See e.g., Limousin+2005 Eq. 10
    
    Inputs:
    vdisp - velocity dispersion of galaxy to set the norm of the mass distribution model (Astropy units expected)
    rcore - PIEMD core radius  of galaxy (Astropy distance units expected)
    rcut - PIEMD cut radius  of galaxy (Astropy distance units expected)
    
    Returns:
    m_piemd_tot - total integrated mass (Msun)
    """
    
    m_piemd_tot = (np.pi * vdisp**2 / G) * rcut**2 / (rcut + rcore)
    
    return m_piemd_tot.to(u.Msun)

def v_disp_from_m_piemd_2d(m_enc, r, rcore, rcut):
    '''
    Get vdisp (central velocity dispersion) given mass enclosed
    within 2D PIEMD aperture
    See Limousin+2005 Eq. 9
    
    Inputs:
    m_enc - known 2D mass within r (Astropy units)
    r - radius within which mass is known (Astropy distance units)
    rcore - PIEMD core radius  of galaxy (Astropy distance units expected)
    rcut - PIEMD cut radius  of galaxy (Astropy distance units expected)
    
    Returns:
    vdisp - velocity dispersion (km/s)
    '''
    
    vdisp = np.sqrt( m_enc * G * ( rcut - rcore ) 
                    / ( 
                        np.pi * rcut * ( rcut - rcore + np.sqrt( rcore**2 + r**2 ) - np.sqrt( rcut**2 + r**2 ) ) 
                    ) )
    
    return vdisp.to(u.km/u.s)

    