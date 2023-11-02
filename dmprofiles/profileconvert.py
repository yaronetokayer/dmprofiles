import numpy as np

from astropy import units as u

def rho_from_m3d(r, m3d):
    '''
    Function to calculate the 3D mass density given the 3D enclosed mass using 
    one-sided differences at the boundaries and central differences for interior points.

    Inputs:
    r - array of r values (Astropy distance units)
    m3d - 3D enclosed mass at each r value (Astropy mass units)
    order - Order of accuracy (1 for first-order, 2 for second-order)

    Returns:
    rho - 3D mass density at each r value (Astropy units)
    '''
    # Volume
    v = 4 * np.pi * r**3 / 3

    # Take derivative 
    rho = np.gradient(m3d, v)

    return rho.to(u.Msun / u.kpc**3)

def sigma_from_m2d(r, m2d):
    '''
    Function to calculate the 2D mass density given the 2D enclosed mass using 
    one-sided differences at the boundaries and central differences for interior points.

    Inputs:
    r - array of r values (Astropy distance units)
    m2d - 2D enclosed mass at each r value (Astropy mass units)

    Returns:
    sigma - 2D mass density at each r value (Astropy units)
    '''
    # Volume
    a = np.pi * r**2

    # Take derivative 
    sigma = np.gradient(m2d, a)

    return sigma.to(u.Msun / u.kpc**2)

def sigma_from_rho(r_array, rho):
    '''
    Function to convert 3D mass density to projected 2D surface mass density, assuming spherical symmetry

    Inputs:
    r_array - array of r values (Astropy distance units)
    rho - array of 3D mass density values (Astropy units)

    Returns:
    sigma - array of projected 2D surface mass density values (Astropy units)
    '''
    
    # Calculate zmax values for all r values
    zmax_array = np.sqrt(r_array[-1]**2 - r_array**2)
    
    # Create the z_array only once
    z_array = np.linspace(0, zmax_array.max(), 10000)
    
    # Interpolate rho values to the z_array
    interpolated_rho = np.interp(np.sqrt(r_array[:, np.newaxis]**2 + z_array**2), r_array, rho)
    
    # Perform numerical integration
    integrand = interpolated_rho.to(u.Msun / u.kpc**3).value
    sigma = 2 * np.trapz(integrand, x=z_array.to(u.kpc).value, axis=1)
    
    # Convert to final units
    sigma = sigma * u.Msun / u.kpc**2
    
    return sigma

### OLD VERSION ###

# def sigma_from_rho(r_array, rho):
#     '''
#     Function to convert 3D mass density to projected 2D surface mass density, assuming spherical symmetry

#     Inputs:
#     r_array - array of r values (Astropy distance units)
#     rho - array of 3D mass density values (Astropy units)

#     Returns:
#     sigma - array of projected 2D surface mass density values (Astropy units)
#     '''
    
#     # Calculate sigma at each r
#     sigma = np.empty(r_array.size)
    
#     for i, r in enumerate(r_array):
#         zmax = np.sqrt(r_array[-1]**2 - r**2)
#         z_array = np.linspace(0, zmax, 10000)
#         integrand = np.interp(np.sqrt(r**2 + z_array**2), r_array, rho).to(u.Msun / u.kpc**3).value

#         sigma[i] = 2 * np.trapz(integrand, x=z_array.to(u.kpc).value)

#     sigma = sigma * u.Msun / u.kpc**2

#     return sigma