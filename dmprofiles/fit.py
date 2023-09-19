'''
Fitting routines for profiles and functional forms
'''
from .profiles_nfw import *
from .profiles_piemd import *
from .profiles_sis import *
from .functionalforms import *
import warnings

#######################################
#### NFW Fitting Functions ############
#######################################

def fit_m_nfw3d(r, m, c_prec=4, r_200_prec=4, c_lims=(1, 101), r_200_lims=(1, 1001) * u.kpc, z=0,
    chatter=False):
    """
    Given data points for 3D spherical mass enclosed at r,
    find c and r_200 to fit NFW profile
    
    Inputs:
    r - array of length 2, radius values (astropy units expected)
    m - array of length 2, enclosed mass values (astropy units expected)
    c_prec - desired precision of c (sigfigs)
    r_200_prec - desired precision of r_200 (sigfigs)
    c_lims - limits of c search
    r_200_lims - limits of r_200 search (astropy units expected)
    z - redshift of the halo
    chatter - option to see fitting progress
    
    Returns:
    c_min - concentration parameter
    r_200_min - virial radius, units consistent with input units
    err_score - log of the squared error for these values of c_min and r_200_min
    flag - True if values are unphysical
    """

    flag = False

    # Booleans to track if we are still fitting
    fitting_c = True; fitting_r_200 = True

    c_res = 1000; r_200_res = 1000
    
    i = 0 # Counter

    while fitting_c or fitting_r_200:
        i += 1

        # Build parameter space
        c_axis = np.linspace(c_lims[0], c_lims[1], c_res)
        r_200_axis = np.linspace(r_200_lims[0], r_200_lims[1], r_200_res)
        cc, r2r2 = np.meshgrid(c_axis, r_200_axis)

        # Calculate squared error
        error = np.zeros((c_res, r_200_res))
        for ind, val in enumerate(r):
            error += (np.log10(m_nfw_3d(val, cc, r2r2, z).value) - np.log10(m[ind].value))**2

        # Find best fit values
        min_ind = np.unravel_index(np.argmin(error, axis=None),error.shape)
        c_min = c_axis[min_ind[1]]
        r_200_min = r_200_axis[min_ind[0]]
        err_score = 100 * 100**( -1 * error[min_ind[0]][min_ind[1]] )

        # If best fit is unphysical, reject fit
        if c_min < 1:
            flag = True
            warnings.warn('parameter c_min is < 1.  This is unphysical.  Being pegged to last physical value.', 
                RuntimeWarning)
            c_min = c_min0 # Set to last kosher value; fitting_c will become false
            if i == 2:
                c_lims = ( 1, 1.1 * c_min0  )
            else:
                c_lims = ( 1, c_min0 + 2 * diff_c ) # Same parameter space as before, but cut off at 1
        if r_200_min < 0:
            flag = True
            warnings.warn('parameter r_200_min is < 0.  This is unphysical.  Being pegged to last physical value.', 
                RuntimeWarning)
            r_200_min = r_200_min0 # Set to last kosher value; fitting_r_200 will become false
            if i == 2:
                r_200_lims = ( 0, 1.1 * r_200_min0 )
            else:
                r_200_lims = ( 0, r_200_min0 + 2 * diff_r_200 ) # Same parameter space as before, but cut off at 0

        # Set new parameter space limits
        if i == 1: # Flags will always be true in first iteration
            c_min0 = c_min
            r_200_min0 = r_200_min
            
            # Best value +/- 10%
            c_lims = ( 0.9 * c_min0, 1.1 * c_min0 )
            r_200_lims = ( 0.9 * r_200_min0, 1.1 * r_200_min0 )
            
        else:
            # Compare with last guess
            diff_c = abs(c_min - c_min0)
            diff_r_200 = abs(r_200_min - r_200_min0)

            if diff_c < abs(c_min0 * 10**(-1 * c_prec)):
                fitting_c = False
            if diff_r_200 < abs(r_200_min0 * 10**(-1 * r_200_prec)):
                fitting_r_200 = False
            
            c_min0 = c_min
            r_200_min0 = r_200_min
            err_score0 = err_score

            if fitting_c:
                c_lims = ( c_min0 - 2 * diff_c, c_min0 + 2 * diff_c )
            if fitting_r_200:
                r_200_lims = ( r_200_min0 - 2 * diff_r_200, r_200_min0 + 2 * diff_r_200 )

        if chatter:      
            print(c_min)
            print(r_200_min)
            print(err_score)
            print('flag: ' + str(flag) )
            print('fitting_c: ' + str(fitting_c) + ', fitting_r_200: ' + str(fitting_r_200))
            print('')

    return round(c_min, sigfigs=c_prec, warn=False), round(r_200_min.value, sigfigs=r_200_prec, warn=False) * u.kpc, round(err_score, decimals=3, warn=False), flag


def fit_m_nfw2d(r, m, c_prec=4, r_200_prec=4, c_lims=(1, 1001), r_200_lims=(1, 1001) * u.kpc, z=0,
               chatter=False):
    """
    Given data points for 2D cylindrical mass enclosed at r,
    find c and r_200 to fit NFW profile
    
    Inputs:
    r - array of min length 2, radius values (astropy units expected)
    m - array of min length 2, enclosed mass values (astropy units expected)
    c_prec - desired precision of c (sigfigs)
    r_200_prec - desired precision of r_200 (sigfigs)
    c_lims - limits of c search
    r_200_lims - limits of r_200 search (astropy units expected)
    z - redshift of the halo
    chatter - option to see fittings progress
    
    Returns:
    c_min - concentration parameter
    r_200_min - virial radius, units consistent with input units
    err_score - log of the squared error for these values of c_min and r_200_min
    flag - True if values are unphysical
    """

    flag = False
    
    # Booleans to track if we are still fitting
    fitting_c = True; fitting_r_200 = True

    c_res = 1000; r_200_res = 1000
    
    i = 0 # Counter

    while fitting_c or fitting_r_200:
        i += 1

        # Build parameter space
        c_axis = np.linspace(c_lims[0], c_lims[1], c_res)
        r_200_axis = np.linspace(r_200_lims[0], r_200_lims[1], r_200_res)
        cc, r2r2 = np.meshgrid(c_axis, r_200_axis)

        # Calculate squared error
        error = np.zeros((c_res, r_200_res))
        for ind, val in enumerate(r):
            error += (np.log10(m_nfw_2d(val, cc, r2r2, z).value) - np.log10(m[ind].value))**2

        # Find best fit values
        min_ind = np.unravel_index(np.argmin(error, axis=None),error.shape)
        c_min = c_axis[min_ind[1]]
        r_200_min = r_200_axis[min_ind[0]]
        err_score = 100 * 100**( -1 * error[min_ind[0]][min_ind[1]] )

        # Set new parameter space limits
        if i == 1: # Flags will always be true in first iteration
            c_min0 = c_min
            r_200_min0 = r_200_min
            
            # Best value +/- 10%
            if 0.9 * c_min0 < 1:
                c_lims = ( 1.000, 1.1 * c_min0 )
            else:
                c_lims = ( 0.9 * c_min0, 1.1 * c_min0 )
            r_200_lims = ( 0.9 * r_200_min0, 1.1 * r_200_min0 )
            
        else:
            # Compare with last guess
            diff_c = abs(c_min - c_min0)
            diff_r_200 = abs(r_200_min - r_200_min0)

            if diff_c < abs(c_min0 * 10**(-1 * c_prec)):
                fitting_c = False
            if diff_r_200 < abs(r_200_min0 * 10**(-1 * r_200_prec)):
                fitting_r_200 = False
            
            c_min0 = c_min
            r_200_min0 = r_200_min

            if fitting_c:
                if c_min0 - 2 * diff_c < 1:
                    c_lims = ( 1.000, c_min0 + 2 * diff_c )
                else:
                    c_lims = ( c_min0 - 2 * diff_c, c_min0 + 2 * diff_c )
            if fitting_r_200:
                if r_200_min0 - 2 * diff_r_200 <= 0:
                    r_200_lims = ( 1e-6 * u.kpc, r_200_min0 + 2 * diff_r_200 )
                else:
                    r_200_lims = ( r_200_min0 - 2 * diff_r_200, r_200_min0 + 2 * diff_r_200 )

        if chatter:
            print(c_min)
            print(r_200_min)
            print(err_score)
            print('fitting_c: ' + str(fitting_c) + ', fitting_r_200: ' + str(fitting_r_200))
            print('')

    # Set flags and warnings if fits are unphysical
    if c_min == 1:
        flag = True
        warnings.warn('best-fit c value is less than 1, which is unphysical. c pinned to 1.  NFW may not be a proper profile.',
            RuntimeWarning)
    if r_200_min == 1e-6 * u.kpc:
        flag = True
        warnings.warn('parameter r_200_min is pinned to minimum value.  NFW may not be a proper profile.',
            RuntimeWarning)

    return round(c_min, sigfigs=c_prec, warn=False), round(r_200_min.value, sigfigs=r_200_prec, warn=False) * u.kpc, round(err_score, decimals=3, warn=False), flag


def fit_m_nfw2d_masscons(r, m, rcut, mcut, rs_prec=5, rs_lims=None, z=0,
                         chatter=False):
    """
    Given data points for 2D cylindrical mass enclosed at r,
    and given a mass conservation constraint,
    find rs and delta_c to fit an NFW profile
    
    Inputs:
    r - array of min length 2, radius values (astropy units expected)
    m - array of min length 2, enclosed mass values (astropy units expected)
    rcut, mcut - rcut and the mass at rcut (astropy distance and mass units, resp.)
    rs_prec - desired precision of rs (sigfigs)
    rs_lims - limits of rs search (astropy distance units)
    z - redshift of the halo
    chatter - option to see fittings progress
    
    Returns:
    rs_min - scale radius in astropy kpc units
    c_min - concentration parameter
    err_score - log of the squared error for this value of rs_min
    flag_mass - True if mass is not conserved due to c being pinned at 1
    """

    if rs_lims is None:
        rs_lims=(0.01, rcut.value) * u.kpc
    
    h = cosmo.H(z) # Hubble parameter
    rho_c = ( 3 * h**2 ) / ( 8 * np.pi * G )

    flag_mass = False

    # Boolean to track if we are still fitting
    fitting = True

    rs_res = 100

    i = 0 # Counter

    while fitting:
        i += 1
        rs_axis = np.geomspace(rs_lims[0], rs_lims[1], rs_res)
        d_c = delta_c_rs(mcut, rcut, rs_axis, rho_c)

        # Calculate squared error
        error = np.zeros(rs_res)
        for ind, val in enumerate(r):
            error += (np.log10(m_nfw_2d_alt(val, d_c, rs_axis, rho_c).value) - np.log10(m[ind].value))**2

        min_ind = np.argmin(error)
        rs_min = rs_axis[min_ind]
        err_score = 100 * 100**( -1 * np.min(error) )

        # Set new parameter space limits
        if i == 1: # Flag will always be true in first iteration
            rs_min0 = rs_min

            # Best value +/- 10%
            rs_lims = ( 0.9 * rs_min0, 1.1 * rs_min0 )

        else:
            # Compare with last guess
            diff_rs = abs(rs_min - rs_min0)

            if diff_rs < abs(rs_min0 * 10**(-1 * rs_prec)):
                fitting = False

            rs_min0 = rs_min

            if fitting:
                if rs_min0 - 2 * diff_rs <= 0:
                    rs_lims = ( 1e-6 * u.kpc, rs_min0 + 2 * diff_rs )
                else:
                    rs_lims = ( rs_min0 - 2 * diff_rs, rs_min0 + 2 * diff_rs )

        if chatter:
            print(rs_min)
            print(err_score)
            print('fitting: ' + str(fitting))
            print('')

    if rs_min == 1e-6 * u.kpc:
        warnings.warn('parameter rs_min is pinned to minimum value. NFW may not be a proper profile.',
            RuntimeWarning)
        
    if chatter:
        print('Deriving c...')
        print('')
    
    d_c = delta_c_rs(mcut, rcut, rs_min, rho_c)

    with warnings.catch_warnings(record=True) as caught_warnings:
        c = invert_delta_c(d_c, chatter=chatter)

    if len(caught_warnings) > 0:
        flag_mass = True
        
    return round(rs_min.value, sigfigs=rs_prec, warn=False) * u.kpc, c, round(err_score, decimals=3, warn=False), flag_mass

########################################
#### tNFW Fitting Functions ############
########################################

def fit_m_tnfw2d(r, m, c_prec=4, r_200_prec=4, tau_prec=4, c_start=10, r_200_start=10 * u.kpc, z=0,
                      chatter=False):
    """
    Given data points for 2D cylindrical mass enclosed at r,
    find c, r_200, and tau to fit tNFW profile
    
    Inputs:
    r - array of min length 2, radius values (astropy units expected)
    m - array of min length 2, enclosed mass values (astropy units expected)
    c_prec - desired precision of c (sigfigs)
    r_200_prec - desired precision of r_200 (sigfigs)
    tau_prec - desired precision of tau (sigfigs)
    c_start - initial value for c_200
    r_200_start - initial value for r_200 (astropy units expected)
    z - redshift of the halo
    chatter - option to see fittings progress
    
    Returns:
    c_min - concentration parameter
    r_200_min - virial radius, units consistent with input units
    tau - dimentionless ratio of truncation radius to scale radius
    err_score - log of the squared error for these values of c_min and r_200_min
    flag - True if values are unphysical
    """

    flag = False
    
    c_lims = (c_start * 0.8, c_start * 1.2)
    r_200_lims = (r_200_start * 0.8, r_200_start * 1.2)
    tau_lims = (0.100, 1000.)
    
    # Booleans to track if we are still fitting
    fitting_c = True; fitting_r_200 = True; fitting_tau = True

    c_res = 100; r_200_res = 100; tau_res = 100
    
    i = 0 # Counter

    while fitting_c or fitting_r_200 or fitting_tau:
        i += 1

        # Build parameter space
        c_axis = np.linspace(c_lims[0], c_lims[1], c_res)
        r_200_axis = np.linspace(r_200_lims[0], r_200_lims[1], r_200_res)
        tau_axis = np.geomspace(tau_lims[0], tau_lims[1], tau_res)
        cc, r2r2, tt = np.meshgrid(c_axis, r_200_axis, tau_axis)

        # Calculate squared error
        error = np.zeros((c_res, r_200_res, tau_res))
        for ind, val in enumerate(r):
            error += (np.log10(m_tnfw_2d(val, cc, r2r2, tt, z).value) - np.log10(m[ind].value))**2

        # Find best fit values
        min_ind = np.unravel_index(np.argmin(error, axis=None),error.shape)
        # Index 0 is r200, 1 is c, 2 is tau
        c_min = c_axis[min_ind[1]]
        r_200_min = r_200_axis[min_ind[0]]
        tau_min = tau_axis[min_ind[2]]
        err_score = 100 * 100**( -1 * error[min_ind[0]][min_ind[1]][min_ind[2]] )

        # Set new parameter space limits

        # For tau, no initial guess
        if i == 1:
            tau_start = tau_min
            
            # Best value +/- 10%
            if 1.1 * tau_start > 1000:
                tau_lims = ( 0.9 * tau_start, 1000. )
            else:
                tau_lims = ( 0.9 * tau_start, 1.1 * tau_start )
                
        else:
            diff_tau = abs(tau_min - tau_start)
            
            if diff_tau < abs(tau_start * 10**(-1 * tau_prec)):
                fitting_tau = False
            
            if fitting_tau:

                tau_start = tau_min
            
                if tau_start - 2 * diff_tau <= 0 and tau_start + 2 * diff_tau > 1000:
                    tau_lims = ( 0.0001, 1000. )
                elif tau_start - 2 * diff_tau <= 0:
                    tau_lims = ( 0.0001, tau_start + 2 * diff_tau )
                elif tau_start + 2 * diff_tau > 1000:
                    tau_lims = ( tau_start - 2 * diff_tau, 1000. )
                else:
                    tau_lims = ( tau_start - 2 * diff_tau, tau_start + 2 * diff_tau )
        
        # Now for c and r_200
        diff_c = abs(c_min - c_start)
        diff_r_200 = abs(r_200_min - r_200_start)

        if diff_c < abs(c_start * 10**(-1 * c_prec)):
            fitting_c = False
        if diff_r_200 < abs(r_200_start * 10**(-1 * r_200_prec)):
            fitting_r_200 = False

        c_start = c_min
        r_200_start = r_200_min

        if fitting_c:
            if c_start - 2 * diff_c < 1:
                c_lims = ( 1.000, c_start + 2 * diff_c )
            else:
                c_lims = ( c_start - 2 * diff_c, c_start + 2 * diff_c )
        if fitting_r_200:
            if r_200_start - 2 * diff_r_200 <= 0:
                r_200_lims = ( 1e-6 * u.kpc, r_200_start + 2 * diff_r_200 )
            else:
                r_200_lims = ( r_200_start - 2 * diff_r_200, r_200_start + 2 * diff_r_200 )

        if chatter:
            print(c_min)
            print(r_200_min)
            print(tau_min)
            print(err_score)
            print('fitting_c: ' + str(fitting_c) + ', fitting_r_200: ' + str(fitting_r_200) + ', fitting_tau: ' + str(fitting_tau))
            print('')

    # Set flags and warnings if fits are unphysical
    if c_min == 1:
        flag = True
        warnings.warn('best-fit c value is less than 1, which is unphysical. c pinned to 1.  NFW may not be a proper profile.',
            RuntimeWarning)
    if r_200_min == 1e-6 * u.kpc:
        flag = True
        warnings.warn('parameter r_200_min is pinned to minimum value.  NFW may not be a proper profile.',
            RuntimeWarning)
    if tau_min == 0.0001:
        flag = True
        warnings.warn('parameter tau is pinned to minimum value 0.0001.  tNFW may not be a proper profile.',
            RuntimeWarning)

    return round(c_min, sigfigs=c_prec, warn=False), round(r_200_min.value, sigfigs=r_200_prec, warn=False) * u.kpc, round(tau_min, sigfigs=tau_prec, warn=False), round(err_score, decimals=3, warn=False), flag

def fit_m_tnfw2d_masscons(r, m, rcut, mcut, rs_prec=4, tau_prec=4,
                          rs_lims=None, tau_lims = (0.010, 1000.),
                          z=0, chatter=False):
    """
    Given data points for 2D cylindrical mass enclosed at r,
    and given a mass conservation constraint,
    find rs, c, and tau to fit a tNFW profile
    
    Inputs:
    r - array of radius values (astropy distance units expected)
    m - array of enclosed mass values (astropy mass units expected)
    rcut, mcut - rcut and the mass at rcut (astropy distance and mass units, resp.)
    rs_prec - desired precision of rs (sigfigs)
    tau_prec - desired precision of tau (sigfigs)
    rs_lims - initial limits of rs search (astropy distance units)
    tau_lims - initial limits of tau search (no units)
    z - redshift of the halo
    chatter - option to see fittings progress
    
    Returns:
    rs_min - scale radius, kpc
    c_min - concentration parameter
    tau_min - ratio of truncation radius to scale radius
    err_score - the error score for these values of c_min and r_200_min
    flag_mass - True if mass is not conserved due to c being pinned at 1
    """

    if rs_lims is None:
        rs_lims=(0.01, rcut.value) * u.kpc
    
    h = cosmo.H(z) # Hubble parameter
    rho_c = ( 3 * h**2 ) / ( 8 * np.pi * G )
    
    flag_mass = False
    
    # Booleans to track if we are still fitting
    fitting_rs = True; fitting_tau = True

    rs_res = 100; tau_res = 100
    
    first_iter = True # Counter

    while fitting_rs or fitting_tau:

        # Build parameter space
        rs_axis = np.geomspace(rs_lims[0], rs_lims[1], rs_res)
        tau_axis = np.geomspace(tau_lims[0], tau_lims[1], tau_res)
        rsrs, tt = np.meshgrid(rs_axis, tau_axis)
        
        d_c = delta_c_rs_tau(mcut, rcut, rsrs, tt, rho_c)

        # Calculate squared error
        error = np.zeros((rs_res, tau_res))
        for ind, val in enumerate(r):
            error += (
                np.log10(m_tnfw_2d_alt(val, d_c, rsrs, tt, rho_c).to(u.Msun).value) 
                - np.log10(m[ind].to(u.Msun).value)
                )**2
        
        # Find best fit values
        min_ind = np.unravel_index(np.argmin(error, axis=None),error.shape)
        # Index 0 is tau, 1 is rs
        rs_min = rs_axis[min_ind[1]]
        tau_min = tau_axis[min_ind[0]]
        
        err_score = 100 * 100**( -1 * np.min(error) )

        # Set new parameter space limits
        if first_iter: # Flags will always be true in first iteration
            rs_min0 = rs_min
            tau_min0 = tau_min
            
            # Best value +/- 10%
            rs_lims = ( 0.9 * rs_min0, 1.1 * rs_min0 )
            if 1.1 * tau_min0 > 1000:
                tau_lims = ( 0.9 * tau_min0, 1000. )
            else:
                tau_lims = ( 0.9 * tau_min0, 1.1 * tau_min0 )

            first_iter = False
                
        else:
            # Compare with last guess
            diff_rs = abs(rs_min - rs_min0)
            diff_tau = abs(tau_min - tau_min0)
            
            if diff_rs < abs(rs_min0 * 10**(-1 * rs_prec)):
                fitting_rs = False
            if diff_tau < abs(tau_min0 * 10**(-1 * tau_prec)):
                fitting_tau = False
            
            rs_min0 = rs_min
            tau_min0 = tau_min
            
            if fitting_rs:
                if rs_min0 - 2 * diff_rs <= 0:
                    rs_lims = ( 1e-6 * u.kpc, rs_min0 + 2 * diff_rs )
                else:
                    rs_lims = ( rs_min0 - 2 * diff_rs, rs_min0 + 2 * diff_rs )
            
            if fitting_tau:
                if tau_min0 - 2 * diff_tau <= 0 and tau_min0 + 2 * diff_tau > 1000:
                    tau_lims = ( 0.0001, 1000. )
                elif tau_min0 - 2 * diff_tau <= 0:
                    tau_lims = ( 0.0001, tau_min0 + 2 * diff_tau )
                elif tau_min0 + 2 * diff_tau > 1000:
                    tau_lims = ( tau_min0 - 2 * diff_tau, 1000. )
                else:
                    tau_lims = ( tau_min0 - 2 * diff_tau, tau_min0 + 2 * diff_tau )

        if chatter:
            print(rs_min)
            print(tau_min)
            print(err_score)
            print('fitting_rs: ' + str(fitting_rs) + ', fitting_tau: ' + str(fitting_tau))
            print('')

    # Set flags and warnings if fits are unphysical
    if rs_min == 1e-6 * u.kpc:
        warnings.warn('parameter rs_min is pinned to minimum value.  NFW may not be a proper profile.',
            RuntimeWarning)
    if tau_min == 0.0001:
        warnings.warn('parameter tau is pinned to minimum value 0.0001.  tNFW may not be a proper profile.',
            RuntimeWarning)

    if chatter:
        print('Deriving c...')
        print('')
        
    d_c = delta_c_rs_tau(mcut, rcut, rs_min, tau_min, rho_c)
    
    with warnings.catch_warnings(record=True) as caught_warnings:
        c = invert_delta_c(d_c, chatter=chatter)

    if len(caught_warnings) > 0:
        flag_mass = True

    return round(rs_min.value, sigfigs=rs_prec, warn=False) * u.kpc, c, round(tau_min, sigfigs=tau_prec, warn=False), round(err_score, decimals=3, warn=False), flag_mass

########################################
#### PIEMD Fitting Function ############
########################################

def fit_m_piemd2d(r, m, vdisp_prec=4, r_prec=4, err_prec=8,
                  vdisp_lims=(10, 1000) * (u.km/u.s), rcore_lims=(.1, 100) * u.kpc, rcut_lims=(1, 10000) * u.kpc,
                  z=0, chatter=False):
    """
    Given data points for 2D cylindrical mass enclosed at r,
    find vdisp, rcore, and rcut to fit PIEMD profile
    
    Inputs:
    r - array of min length 3, radius values (kpc)
    m - array of min length 3, enclosed mass values (Msun)
    vdisp_prec - desired precision of vdisp (sigfigs)
    r_prec - desired precision of r_core and r_cut (sigfigs)
    err_prec - precision of the err_score that overrides the parameter precision for runaway cases
    vdisp_lims - limits of vdisp search (km/s)
    rcore_lims - limits of rcore search (kpc)
    rcut_lims - limits of rcut search (kpc)
    z - redshift of the halo
    
    Returns:
    c_min - concentration parameter
    r_200_min - virial radius, units consistent with input units
    err_score - log of the squared error for these values of c_min and r_200_min
    """
    
    # Booleans to track if we are still fitting
    fitting_vdisp = True; fitting_rcore = True; fitting_rcut = True

    vdisp_res = 500; r_res = 500
    
    # Stage 1: fit rcore and rcut to match the curve shape
    
    if chatter:
        print('STAGE 1: Fitting slope')
    
    i = 0 # Counter

    while fitting_rcore or fitting_rcut:
        i += 1

        # Build parameter space
        rcore_axis = np.linspace(rcore_lims[0], rcore_lims[1], r_res)
        rcut_axis = np.linspace(rcut_lims[0], rcut_lims[1], r_res)
        rrcore, rrcut = np.meshgrid(rcore_axis, rcut_axis)

        # Calculate shape score
        # Assume vdisp = 100 km/s
        errs = []
        for ind, val in enumerate(r):
            errs.append( np.log10(m_piemd_2d(val, 100 * u.km/u.s, rrcore, rrcut, z).value) - np.log10(m[ind].value) )
        
        shape_scores = np.std(errs, axis=0)

        # Find best fit values
        min_ind = np.unravel_index(np.argmin(shape_scores, axis=None),shape_scores.shape)
        # Index 0 is rcut, 1 is rcore
        rcore_min = rcore_axis[min_ind[1]]
        rcut_min = rcut_axis[min_ind[0]]
        shape_score = shape_scores[min_ind[0]][min_ind[1]]

        # Set new parameter space limits
        if i == 1: # Flags will always be true in first iteration
            rcore_min0 = rcore_min
            rcut_min0 = rcut_min
            shape_score0 = shape_score
            
            # Best value +/- 10%
            rcore_lims = ( 0.9 * rcore_min0, 1.1 * rcore_min0 )
            rcut_lims = ( 0.9 * rcut_min0, 1.1 * rcut_min0 )
            
        else:
            # Compare with last guess
            diff_rcore = abs(rcore_min - rcore_min0)
            diff_rcut = abs(rcut_min - rcut_min0)
            diff_shape = abs(shape_score - shape_score0)

            if diff_rcore < abs(rcore_min0 * 10**(-1 * r_prec)):
                fitting_rcore = False
            if diff_rcut < abs(rcut_min0 * 10**(-1 * r_prec)):
                fitting_rcut = False
            
            # Set cutoff for rcut at 1000 kpc
            if rcut_min >= 1000 * u.kpc:
                rcut_min = 1000 * u.kpc
                fitting_rcut = False
                
            rcore_min0 = rcore_min
            rcut_min0 = rcut_min
            shape_score0 = shape_score
            
            if fitting_rcore:
                rcore_lims = ( rcore_min0 - 2 * diff_rcore, rcore_min0 + 2 * diff_rcore )
            if fitting_rcut:
                rcut_lims = ( rcut_min0 - 2 * diff_rcut, rcut_min0 + 2 * diff_rcut )
        
        if chatter:
            print(min_ind)
            print(rcore_min)
            print(rcut_min)
            print(shape_score)
            print('fitting_rcore: ' + str(fitting_rcore) + ', fitting_rcut: ' + str(fitting_rcut))
            print('')
            
    # Stage 2: fit vdisp
    
    if chatter:
        print('STAGE 2: Fitting normalization')
    
    i = 0 # Counter

    while fitting_vdisp:
        i += 1

        # Build parameter space
        vdisp_axis = np.linspace(vdisp_lims[0], vdisp_lims[1], vdisp_res)

        # Calculate squared error
        error = np.zeros(vdisp_res)
        for ind, val in enumerate(r):
            error += (np.log10(m_piemd_2d(val, vdisp_axis, rcore_min, rcut_min, z).value) - np.log10(m[ind].value))**2

        # Find best fit values
        min_ind = np.argmin(error)
        vdisp_min = vdisp_axis[min_ind]
        err_score = 100 * 100**(-1 * error[min_ind])

        # Set new parameter space limits
        if i == 1: # Flags will always be true in first iteration
            vdisp_min0 = vdisp_min
            
            # Best value +/- 10%
            vdisp_lims = ( 0.9 * vdisp_min0, 1.1 * vdisp_min0 )
            
        else:
            # Compare with last guess
            diff_vdisp = abs(vdisp_min - vdisp_min0)

            if diff_vdisp < abs(vdisp_min0 * 10**(-1 * vdisp_prec)):
                fitting_vdisp = False
                
            vdisp_min0 = vdisp_min
            
            if fitting_vdisp:
                vdisp_lims = ( vdisp_min0 - 2 * diff_vdisp, vdisp_min0 + 2 * diff_vdisp )
        
        if chatter:
            print(min_ind)
            print(vdisp_min)
            print(err_score)
            print('fitting_vdisp: ' + str(fitting_vdisp))
            print('')

    return round(vdisp_min.value, sigfigs=vdisp_prec, warn=False) * u.km / u.s, round(rcore_min.value, sigfigs=r_prec, warn=False) * u.kpc, round(rcut_min.value, sigfigs=r_prec, warn=False) * u.kpc, round(err_score, decimals=3, warn=False)

#########################################
#### Other Fitting Functions ############
#########################################

def fit_powerlaw(x, y, k_prec=3, alpha_prec=4, k_lims=(10, 10**10), alpha_lims=(-2, -0.01), 
                 chatter=False):
    """
    Given data points, find a best fit powerlaw, y = k * (x**alpha)
    
    Inputs:
    x - independent variable data
    y - dependent variable data
    k_prec - desired precision of k (sigfigs)
    alpha_prec - desired precision of alpha (sigfigs)
    k_lims - limits of initial k search
    alpha_lims - limits of inital alpha search
    chatter - option to see outputs
    
    Returns:
    k_min - concentration parameter
    alpha_200_min - virial radius, units consistent with input units
    err_score - log of the squared error for these values of c_min and r_200_min
    """
    
    # Booleans to track if we are still fitting
    fitting_k = True; fitting_alpha = True

    k_res = 1000; alpha_res = 1000
    
    # Stage 1: fit alpha
    
    if chatter:
        print('STAGE 1: Fitting slope')
    
    i = 0 # Counter
    
    while fitting_alpha:
        i += 1
        
        # Build 1D parameter space
        alpha_axis = np.linspace(alpha_lims[0], alpha_lims[1], alpha_res)
        
        # Calculate shape score
        # Assume k = 1000
        error = []
        for ind, val in enumerate(x):
            error.append( np.log10(powerlaw(val, 1000, alpha_axis)) - np.log10(y[ind]) )
            
        shape_scores = np.std(error, axis=0)
        
        # Find best fit value
        min_ind = np.argmin(shape_scores)
        alpha_min = alpha_axis[min_ind]
        shape_score = shape_scores[min_ind]
        
        # Set new parameter space limits
        if i == 1: # Flag will always be true in first iteration
            alpha_min0 = alpha_min
            shape_score0 = shape_score
            
            # Best value +/- 10%
            alpha_lims = ( 0.9 * alpha_min0, 1.1 * alpha_min0 )
            
        else:
            # Compare with last guess
            diff_alpha = abs(alpha_min - alpha_min0)
            diff_shape = abs(shape_score - shape_score0)

            if diff_alpha < abs(alpha_min0 * 10**(-1 * alpha_prec)):
                fitting_alpha = False
            
            alpha_min0 = alpha_min
            shape_score0 = shape_score
            
            if fitting_alpha:
                alpha_lims = ( alpha_min0 - 2 * diff_alpha, alpha_min0 + 2 * diff_alpha )
                
        if chatter:
            print(min_ind)
            print(alpha_min)
            print(shape_score)
            print('fitting_alpha: ' + str(fitting_alpha))
            print('')
            
    # Stage 2: fit k
    
    if chatter:
        print('STAGE 2: Fitting normalization')
        
    i = 0 # Counter
                
    while fitting_k:
        i += 1

        # Build parameter space
        k_axis = np.linspace(k_lims[0], k_lims[1], k_res)

        # Calculate squared error
        error = np.zeros(k_res)
        for ind, val in enumerate(x):
            error += (powerlaw(val, k_axis, alpha_min) - y[ind])**2

        # Find best fit values
        min_ind = np.argmin(error)
        k_min = k_axis[min_ind]
        err_score = 100 * (-1 * error[min_ind])

        # Set new parameter space limits
        if i == 1: # Flags will always be true in first iteration
            k_min0 = k_min
            
            # Best value +/- 10%
            k_lims = ( 0.9 * k_min0, 1.1 * k_min0 )
            
        else:
            # Compare with last guess
            diff_k = abs(k_min - k_min0)

            if diff_k < abs(k_min0 * 10**(-1 * k_prec)):
                fitting_k = False
            
            k_min0 = k_min
            
            if fitting_k:
                k_lims = ( k_min0 - 2 * diff_k, k_min0 + 2 * diff_k )
        
        if chatter:
            print(min_ind)
            print(k_min)
            print(error[min_ind])
            print(err_score)
            print('kfit: ' + str(fitting_k))
            print('')

    return round(k_min, sigfigs=k_prec, warn=False), round(alpha_min, sigfigs=alpha_prec, warn=False), err_score

def invert_delta_c(target_delta_c, tol=1e-6, a=1.0, b=10000, max_iterations=100, chatter=False):
    """
    Find the concentration parameter c given a target value of delta_c using the bisection method.
    
    Inputs:
    target_delta_c - the desired value of delta_c
    tol - tolerance for the root approximation (optional, default is 1e-6)
    a, b - endpoints of the initial search interval (optional, default is [1.0, 10000])
    max_iterations - maximum number of iterations (optional, default is 100)
    chatter - option to output progress
    
    Returns:
    c - concentration parameter
    """
    if (delta_c(a) - target_delta_c) * (delta_c(b) - target_delta_c) >= 0:
        if (delta_c(a) - target_delta_c) >= 0:
            if (delta_c(a) - target_delta_c) > 0:
                warnings.warn('best-fit c value is less than 1. c pinned to 1.', 
                    RuntimeWarning)
            return round(a, decimals=int(-np.log10(tol)), warn=False)
        elif (delta_c(b) - target_delta_c) <= 0:
            if (delta_c(b) - target_delta_c) < 0:
                warnings.warn('best-fit c value is greater than b. c pinned to ' + str(b), 
                    RuntimeWarning)
            return round(b, decimals=int(-np.log10(tol)), warn=False)

    if a < 1.0:
        a = 1.0

    for iteration in range(max_iterations):
        c = (a + b) / 2
        f_c = delta_c(c)

        if chatter:
            print(c)

        if abs(f_c - target_delta_c) < tol:
            return round(c, decimals=int(-np.log10(tol)), warn=False)

        if (delta_c(a) - target_delta_c) * (f_c - target_delta_c) < 0:
            b = c
        else:
            a = c

        if a >= b:
            raise ValueError("Interval [a, b] collapsed during the bisection process.")

    raise ValueError("Bisection method did not converge within the given number of iterations.")

# def invert_delta_c(d_c, c_prec=5, c_lims=(1, 1001), chatter=False):
#     '''
#     Fitting algorithm to invert delta_c
#     Needed for mass conserved nfw fit
#     '''
    
#     flag = False
    
#     fitting = True
    
#     c_res = 100
    
#     i = 0 # Counter
    
#     while fitting:
#         i +=1 
        
#         c_axis = np.linspace(c_lims[0], c_lims[1], c_res)
        
#         # Calculate squared error
#         error = (delta_c(c_axis) - d_c)**2
        
#         min_ind = np.argmin(error)
#         c_min = c_axis[min_ind]
        
#         # Set new parameter space limits
#         if i == 1: # Flag will always be true in first iteration
#             c_min0 = c_min

#             # Best value +/- 10%
#             if 0.9 * c_min0 < 1:
#                 c_lims = ( 1.000, 1.1 * c_min0 )
#             else:
#                 c_lims = ( 0.9 * c_min0, 1.1 * c_min0 )
            
#         else:
#             # Compare with last guess
#             diff_c = abs(c_min - c_min0)

#             if diff_c < abs(c_min0 * 10**(-1 * c_prec)):
#                 fitting = False

#             c_min0 = c_min

#             if fitting:
#                 if c_min0 - 2 * diff_c < 1:
#                     c_lims = ( 1.0000, c_min0 + 2 * diff_c )
#                 else:
#                     c_lims = ( c_min0 - 2 * diff_c, c_min0 + 2 * diff_c )
                    
#         if chatter:
#             print(c_min)
#             print('fitting: ' + str(fitting))
#             print('')
            
#     if c_min == 1:
#         flag = True
#         warnings.warn('parameter c_min is 1.  Check parameter space limits, or NFW may not be a proper profile.')
        
#     return round(c_min, sigfigs=c_prec), flag