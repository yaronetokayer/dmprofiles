'''
Fitting routines for profiles and functional forms
'''
from .profiles import *
from .functionalforms import *

#######################################
#### NFW Fitting Functions ############
#######################################

def fit_m_nfw3d(r, m, c_prec=4, r_200_prec=4, c_lims=(1, 101), r_200_lims=(1, 1001) * u.kpc, z=0,
    chatter=False):
    """
    Given data points for 3D spherical mass enclosed at r,
    find c and r_200 to fit NFW profile
    
    Inputs:
    r - array of length 2, radius values (kpc)
    m - array of length 2, enclosed mass values (Msun)
    c_prec - desired precision of c (sigfigs)
    r_200_prec - desired precision of r_200 (sigfigs)
    c_lims - limits of c search
    r_200_lims - limits of r_200 search (kpc)
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
            warnings.warn('parameter c_min is < 1.  This is unphysical.  Being pegged to last physical value.')
            c_min = c_min0 # Set to last kosher value; fitting_c will become false
            if i == 2:
                c_lims = ( 1, 1.1 * c_min0  )
            else:
                c_lims = ( 1, c_min0 + 2 * diff_c ) # Same parameter space as before, but cut off at 1
        if r_200_min < 0:
            flag = True
            warnings.warn('parameter r_200_min is < 0.  This is unphysical.  Being pegged to last physical value.')
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

    return round(c_min, sigfigs=c_prec), round(r_200_min.value, sigfigs=r_200_prec) * u.kpc, round(err_score, decimals=3), flag


def fit_m_nfw2d(r, m, c_prec=4, r_200_prec=4, c_lims=(1, 1001), r_200_lims=(1, 1001) * u.kpc, z=0,
               chatter=False):
    """
    Given data points for 2D cylindrical mass enclosed at r,
    find c and r_200 to fit NFW profile
    
    Inputs:
    r - array of min length 2, radius values (kpc)
    m - array of min length 2, enclosed mass values (Msun)
    c_prec - desired precision of c (sigfigs)
    r_200_prec - desires precision of r_200 (sigfigs)
    c_lims - limits of c search
    r_200_lims - limits of r_200 search (kpc)
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

        # If best fit is unphysical, reject fit
        if c_min < 1:
            flag = True
            warnings.warn('parameter c_min is < 1.  This is unphysical.  Returning to last physical value.')
            c_min = c_min0 # Set to last kosher value; fitting_c will become false
            if i == 2:
                c_lims = ( 1, 1.1 * c_min0  )
            else:
                c_lims = ( 1, c_min0 + 2 * diff_c ) # Same parameter space as before, but cut off at 1
        if r_200_min < 0:
            flag = True
            warnings.warn('parameter r_200_min is < 0.  This is unphysical.  Returning to last physical value.')
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

    return round(c_min, sigfigs=c_prec), round(r_200_min.value, sigfigs=r_200_prec) * u.kpc, round(err_score, decimals=3), flag

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

    return round(vdisp_min.value, sigfigs=vdisp_prec) * u.km / u.s, round(rcore_min.value, sigfigs=r_prec) * u.kpc, round(rcut_min.value, sigfigs=r_prec) * u.kpc, round(err_score, decimals=3)

###########################################
#### Powerlaw Fitting Function ############
###########################################

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

    return round(k_min, sigfigs=k_prec), round(alpha_min, sigfigs=alpha_prec), err_score