import numpy as np

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.constants import G


# Define a default cosmology

cosmo = FlatLambdaCDM(H0=69, Om0=0.3)

######################
#### NFW Profiles ####
######################

def rho_nfw(r, c, r_200, z=0, cosmo=cosmo):
    """
    NFW 3D density at r, given the concentration parameter and the scale radius
    cgs units
    
    Inputs:
    r - array-like, radius at which to compute the mass density
    c - concentration parameter
    r_200 - the virial radius (unit consistent with r)
    z - redshift of the halo (default is 0)
    cosmo - astropy cosmology class instantiation.
            Default is FlatLambdaCDM(H0=69, Om0=0.3)
    
    Returns:
    rho_nfw - 3D density at r (g / cm**3)
    """
    
    d_c = delta_c(c)
    
    h = cosmo.H(z) # Hubble parameter
    
    rho_c = ( 3 * h**2 ) / ( 8 * np.pi * G )
    
    r_s = r_200 / c
    
    return ( ( d_c * rho_c ) / ( (r / r_s) * ( 1 + ( r / r_s ) )**2 ) ).to(u.g / u.cm**3)

def m_nfw_3d(r, c, r_200, z=0, cosmo=cosmo):
    """
    NFW 3D spherical mass enclosed at r, given the concetration parameter and the scale radius
    cgs units
    
    Inputs:
    r - radius at which to compute the enclosed mass (astropy units expected)
    c - concentration parameter
    r_200 - array-like, radius of the halo inside which the mass density is 200*rho_c
           astropy units expected
    z - redshift of the halo (default is 0)
    cosmo - astropy cosmology class instantiation.
            Default is FlatLambdaCDM(H0=69, Om0=0.3)
    
    Returns:
    m_nfw_3d - 3D spherical integrated mass at r (Msun)
    """
    
    d_c = delta_c(c)
    
    h = cosmo.H(z) # Hubble parameter
    
    rho_c = ( 3 * h**2 ) / ( 8 * np.pi * G )
    
    r_s = r_200 / c  
    
    x = r / r_s
    
    m_nfw_3d = 4 * np.pi * d_c * rho_c * r_s**3 * ( np.log( 1 + x ) - ( x / ( 1 + x ) ) )
    
    return m_nfw_3d.to(u.Msun)

def m_nfw_2d(r, c, r_200, z=0, cosmo=cosmo):
    """
    NFW 2D cylindrical mass enclosed at r, given the concentration parameter and r_200
    
    Inputs:
    r - radius at which to compute the enclosed mass (astropy units expected)
    c - concentration parameter
    r_200 - array-like, radius of the halo inside which the mass density is 200*rho_c
           astropy units expected
    z - redshift of the halo (default is 0)
    cosmo - astropy cosmology class instantiation.
            Default is FlatLambdaCDM(H0=69, Om0=0.3)
    
    Returns:
    m_nfw_2d - 2D cylindrical integrated mass at r (Msun)
    """
    
    d_c = delta_c(c)
    
    h = cosmo.H(z) # Hubble parameter
    
    rho_c = ( 3 * h**2 ) / ( 8 * np.pi * G )
    
    r_s = r_200 / c
    
    x = ( r / r_s ).value # x is dimensionless
    
    # x = 1
    # returns zero for all other values
    sigma_1 = np.where(x == 1, x, np.zeros(x.shape)) * 4 * r_s * d_c * rho_c * ( 1 + np.log(1 / 2) )    
    
    # x < 1
    # returns zero for all other values
    sigma_l = ( (4 / x**2) * r_s * d_c * rho_c * 
                ( ( 2 / np.sqrt(1 - x**2, out=np.ones(x.shape), where=x < 1) ) 
                 * np.arctanh( np.sqrt( (1 - x) / (1 + x), out=np.zeros(x.shape),where=x<1) )
                 + np.log(x / 2, out=np.zeros(x.shape), where=x < 1) ) ) 
    
    # x > 1
    # returns zero for all other values
    sigma_g = ( (4 / x**2) * r_s * d_c * rho_c 
               * ( ( 2 / np.sqrt(x**2 - 1, out=np.ones(x.shape), where=x > 1) ) 
                  * np.arctan( np.sqrt( (x - 1) / (1 + x), out=np.zeros(x.shape), where=x > 1) )
                  + np.log(x / 2, out=np.zeros(x.shape), where=x > 1) ) )
    
    # Convert from avg surface density to total enclosed mass
    m_nfw_2d = np.pi * r**2 * ( sigma_l + sigma_1 + sigma_g )
    
    return m_nfw_2d.to(u.Msun)

def m200_nfw(r_200, z=0, cosmo=cosmo):
    """
    3D mass of an NFW halo contained within a radius of r200.
    This is equivalent to passing r200 to m_nfw_3d, but the equation simplifies at r200.
    See, e.g., Wright and Brainerd (2000)
    
    Inputs:
    r200 - array-like, radius of the halo inside which the mass density is 200*rho_c
           astropy units expected
    z - redshift of the halo (default is 0)
    cosmo - astropy cosmology class instantiation.
            Default is FlatLambdaCDM(H0=69, Om0=0.3)
    
    Returns:
    m200 - mass enclosed within r200 (Msol)
    """
    
    h = cosmo.H(z) # Hubble parameter
    
    rho_c = ( 3 * h**2 ) / ( 8 * np.pi * G )
    
    return ( ( 800 * np.pi / 3 ) * rho_c * r_200**3 ).to(u.Msun)

def delta_c(c):
    """
    Compute delta_c, the characteristic overdensity for a halo in the NFW model
    
    Inputs:
    c - concentration parameter
    
    Returns:
    delta_c - characteristic overdensity
    """
        
    return ( 200 / 3 ) * ( c**3 / ( np.log(1 + c) - ( c / (1 + c) ) ) )

#######################
#### tNFW Profiles ####
#######################
'''
The truncated NFW profile, as presented in in Baltz, Marshall, Oguri (2009)
'''

def rho_tnfw(r, c, r_200, tau, z=0, cosmo=cosmo):
    """
    tNFW 3D density at r, with the concentration parameter, scale radius, and tau=r_t/r_s
    as free parameters.
    Returns in cgs units
    
    Inputs:
    r - array-like, radius at which to compute the mass density (must include astropy units)
    c - concentration parameter
    r_200 - array-like, radius of the halo inside which the mass density is 200*rho_c
            astropy units expected
    tau - ratio of truncation radius to scale radius
    z - redshift of the halo (default is 0)
    cosmo - astropy cosmology class instantiation.
            Default is FlatLambdaCDM(H0=69, Om0=0.3)
    
    Returns:
    rho_nfw - 3D density at r (g / cm**3)
    """
    
    r_s = r_200 / c
    
    return ( 
        rho_nfw(r, c, r_200, z=z, cosmo=cosmo) * ( tau**2 / ( tau**2 + ( r / r_s )**2 ) ) 
        ).to(u.g / u.cm**3)

def m_tnfw_2d(r, c, r_200, tau, z=0, cosmo=cosmo):
    """
    Truncated NFW 2D cylindrical mass enclosed at r, given the concentration parameter, 
    the r_200, and tau = r_t/r_s.
    
    Inputs:
    r - radius at which to compute the enclosed mass (must include astropy units)
    c - concentration parameter
    r_200 - array-like, radius of the halo inside which the mass density is 200*rho_c
            astropy units expected
    tau - ratio of truncation radius to scale radius
    z - redshift of the halo (default is 0)
    cosmo - astropy cosmology class instantiation.
            Default is FlatLambdaCDM(H0=69, Om0=0.3)
    
    Returns:
    m_tnfw_2d - 2D cylindrical integrated mass at r (Msun)
    """
    
    r_s = r_200 / c
    
    x = ( r / r_s ).value
    
    m0 = _m0(c, r_s, z=z, cosmo=cosmo)
    
    m_tnfw_2d = ( m0 * ( tau**2 / ( tau**2 + 1 )**2 ) * 
                 ( ( tau**2 + 1 + 2 * ( x**2 - 1 ) ) * _f(x) + tau * np.pi + 
                 ( tau**2 - 1 ) * np.log(tau) + 
                  np.sqrt( tau**2 + x**2 ) * ( -np.pi + ( ( tau**2 - 1 ) / tau ) * _l(x, tau) ) )
                )
    
    return m_tnfw_2d.to(u.Msun)

def m_tot_tnfw(c, r_200, tau, z=0, cosmo=cosmo):
    """
    Total mass in truncated NFW halo
    
    Inputs:
    c - concentration parameter
    r_200 - array-like, radius of the halo inside which the mass density is 200*rho_c
            astropy units expected
    tau - ratio of truncation radius to scale radius
    z - redshift of the halo (default is 0)
    cosmo - astropy cosmology class instantiation.
            Default is FlatLambdaCDM(H0=69, Om0=0.3)
    
    Returns:
    m_tot_tnfw - in units of Msun
    """
    
    m0 = _m0(c, r_200 / c, z=z, cosmo=cosmo)

    return ( m0 * ( tau**2 / ( tau**2 + 1 )**2 ) * 
            ( ( tau**2 - 1 ) * np.log( tau ) + tau * np.pi - (tau**2 + 1 ) ) ).to(u.Msun)

def _m0(c, r_s, z=0, cosmo=cosmo):
    """
    M_0 as defined in Baltz, Marshall, Oguri (2009)
    Mass normalization in NFW profiles
    
    Inputs:
    c - concentration parameter
    r_s - r_200 / c, the scale radius (unit consistent with r)
    z - redshift of the halo (default is 0)
    cosmo - astropy cosmology class instantiation.
            Default is FlatLambdaCDM(H0=69, Om0=0.3)
    
    Returns:
    M_0(x) - Mass normalization
    """
    
    d_c = delta_c(c)
    
    h = cosmo.H(z) # Hubble parameter
    
    rho_c = ( 3 * h**2 ) / ( 8 * np.pi * G )
    
    return 4 * np.pi * r_s**3 * d_c * rho_c

def _f(x):
    """
    F(x) as defined in Baltz, Marshall, Oguri (2009)
    For purely imaginary numerator and denominator, branches are chosen such that F(x) remains > 0
    
    Inputs:
    x - array-like, must be greater than 0
    
    Returns:
    F(x)
    """
    
    if not isinstance(x, np.ndarray):
        x = np.array([x])
    
    # x = 1
    # returns zero for all other values
    f_1 = np.where(x == 1, x, np.zeros(x.shape))  
    
    x = x.astype('complex')
    
    # x < 1
    # returns zero for all other values
    f_l = ( -np.arccos( 1 / x, out=np.zeros(x.shape, dtype='complex'), where=x.real < 1 ) / 
           np.sqrt( x**2 - 1, out=np.ones(x.shape, dtype='complex'), where=x.real < 1 ) ).real
    
    # x > 1
    # returns zero for all other values
    f_g = ( np.arccos( 1 / x, out=np.zeros(x.shape, dtype='complex'), where=x.real > 1 ) / 
           np.sqrt( x**2 - 1, out=np.ones(x.shape, dtype='complex'), where=x.real > 1 ) ).real
    
    return f_l + f_1 + f_g

def _l(x, tau):
    """
    L(x) as defined in Baltz, Marshall, Oguri (2009)
    
    Inputs:
    x - array-like, must be greater than 0
    tau - scalar
    
    Returns:
    L(x)
    """
    
    return np.log( x / ( np.sqrt( tau**2 + x**2 ) + tau ) )

########################
#### PIEMD Profiles ####
########################

def m_piemd_2d(r, vdisp, rcore, rcut, z=0):
    """
    PIEMD 2D cylindrical mass enclosed at r, given the velocity dispersion, r_core and r_cut
    
    Inputs:
    r - radius at which to compute the enclosed mass (kpc)
    vdisp - velocity dispersion of galaxy to set the norm of the mass distribution model (km/s)
    rcore - PIEMD core radius  of galaxy (arcsec)
    rcut - PIEMD cut radius  of galaxy (arcsec)
    z - redshift of the halo (default is 0)
    
    Returns:
    m_piemd_2d - 2D cylindrical integrated mass at r (Msun)
    """
    
    sigma_0 = ((vdisp)**2) / ( 2 * G * rcore)
    
    m_piemd_2d = ( ( 2 * np.pi * sigma_0 * rcore * rcut / (rcut - rcore) ) 
                  * ( np.sqrt(rcore**2 + r**2) 
                     - np.sqrt(rcut**2 + r**2) 
                     + rcut - rcore ) )
    
    return m_piemd_2d.to(u.Msun)

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