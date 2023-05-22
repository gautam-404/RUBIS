#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 18:36:42 2022

@author: phoudayer
"""

#%% Modules cell

import time
import numpy        as np
import scipy.sparse as sps
from scipy.interpolate      import CubicHermiteSpline
from scipy.linalg.lapack    import dgbtrf, dgbtrs
from scipy.special          import roots_legendre, eval_legendre

from dotdict                import DotDict
from low_level              import (
    integrate, 
    integrate2D,
    interpolate_func, 
    find_r_eq,
    find_r_pol,
    pl_eval_2D,
    pl_project_2D,
    lagrange_matrix_P,
    plot_f_map
)
from rotation_profiles      import solid, lorentzian, plateau, la_bidouille 
from generate_polytrope     import polytrope

#%% High-level functions cell
    
def set_params() : 
    """
    Function returning the main parameters of the file.
    
    Returns
    -------
    model_choice : string or DotDict instance
        Name of the file containing the 1D model or dictionary containing
        the information requiered to compute a polytrope of given
        index : {
            index : float
                Polytrope index
            surface_pressure : float
                Surface pressure expressed in units of central
                pressure, ex: 1e-12 => P0 = 1e-12 * PC
            radius : float
                Radius of the model
            mass : float
                Mass of the model
            res : integer
                Radial resolution of the model
        }
    rotation_profile : function(r, cth, omega)
        Function used to compute the centrifugal potential and its 
        derivative. Possible choices are {solid, lorentzian, plateau}.
        Explanations regarding this profiles are available in the 
        corresponding functions.
    rotation_target : float
        Target for the rotation rate.
    rate_difference : float
        The rotation rate difference between the centre and equator in the
        cylindrical rotation profile. For instance, rate_difference = 0.0
        would corresond to a solid rotation profile while 
        rate_difference = 0.5 indicates that the star's centre rotates 50%
        faster than the equator.
        Only appears in cylindrical rotation profiles.
    rotation_scale : float
        Homotesy factor on the x = r*sth / Req axis for the rotation profile.
        Only appear in the plateau rotation profile.
    max_degree : integer
        Maximum l degree to be considered in order to do the
        harmonic projection.
    angular_resolution : integer
        Angular resolution for the mapping. Better to take an odd number 
        in order to include the equatorial radius.
    full_rate : integer
        Number of iterations before reaching full rotation rate. As a 
        rule of thumb, ~ 1 + int(-np.log(1-rotation_target)) iterations
        should be enough to ensure convergence (in the solid rotation case!).
    mapping_precision : float
        Precision target for the convergence criterion on the mapping.
    lagrange_order : integer
        Choice of Lagrange polynomial order in integration / interpolation
        routines. 
        2 should be enough.
    spline_order : integer
        Choice of B-spline order in integration / interpolation
        routines. 
        3 is recommanded (must be odd in anycase).
    plot_resolution : integer
        Angular resolution for the mapping plot.
    save_name : string
        Filename in which to scaled model will be saved.

    """
    #### MODEL CHOICE ####
    # model_choice = "1Dmodel_1.97187607_G1.txt"     
    model_choice = DotDict(index=1.0, surface_pressure=0.0, R=1.0, M=1.0, res=10_000)

    #### ROTATION PARAMETERS ####      
    rotation_profile = solid
    # rotation_profile = la_bidouille('rota_eq.txt', smoothing=1e-5)
    rotation_target = 0.089195487 ** 0.5
    # rotation_target = 0.9
    central_diff_rate = 1.0
    rotation_scale = 1.0
    
    #### SOLVER PARAMETERS ####
    max_degree = angular_resolution = 201
    full_rate = 1
    mapping_precision = 1e-10
    lagrange_order = 3
    spline_order = 5
    
    #### OUTPUT PARAMETERS ####
    plot_resolution = 501
    save_name = give_me_a_name(model_choice, rotation_target)
    
    return (
        model_choice, rotation_target, full_rate, rotation_profile,
        central_diff_rate, rotation_scale, mapping_precision, 
        spline_order, lagrange_order, max_degree, 
        angular_resolution, plot_resolution, save_name
    )

def give_me_a_name(model_choice, rotation_target) : 
    """
    Constructs a name for the save file using the model name
    and the rotation target.

    Parameters
    ----------
    model_choice : string or Dotdict instance.
        File name or polytrope caracteristics.
    rotation_target : float
        Final rotation rate on the equator.

    Returns
    -------
    save_name : string
        Output file name.

    """
    radical = (
        'poly_' + str(int(model_choice.index)) 
        if isinstance(model_choice, DotDict) 
        else model_choice.split('.txt')[0]
    )
    save_name = radical + '_deform_' + str(rotation_target) + '.txt'
    return save_name


def init_1D() : 
    """
    Function reading the 1D model file 'MOD_1D' (or generating a
    polytrope if MOD_1D is a dictionary). If additional variables are 
    found in the file, they are left unchanged and returned in the 
    'SAVE' file.

    Returns
    -------
    P0 : float
        Value of the surface pressure after normalisation.
    N : integer
        Radial resolution of the model.
    M : float
        Total mass of the model.
    R : float
        Radius of the model.
    r : array_like, shape (N, ), [GLOBAL VARIABLE]
        Radial coordinate after normalisation.
    zeta : array_like, shape (N, ), [GLOBAL VARIABLE]
        Spheroidal coordinate
    rho : array_like, shape (N, )
        Radial density of the model after normalisation.
    other_var : array_like, shape (N, N_var)
        Additional variables found in 'MOD1D'.

    """
    if isinstance(MOD_1D, DotDict) :        
        # The model properties are user-defined
        N = MOD_1D.res    or 1001
        M = MOD_1D.mass   or 1.0
        R = MOD_1D.radius or 1.0
        
        # Polytrope computation
        model = polytrope(*MOD_1D.values())
        
        # Normalisation
        r   = model.r     /  R
        rho = model.rho   / (M/R**3)
        other_var = np.empty_like(r)
        P0  = model.p[-1] / (G*M**2/R**4)
        
    else : 
        # Reading file 
        surface_pressure, radial_res = np.genfromtxt(
            './Models/'+MOD_1D, max_rows=2, unpack=True
        )
        r1D, rho1D, *other_var = np.genfromtxt(
            './Models/'+MOD_1D, skip_header=2, unpack=True
        )
        _, idx = np.unique(r1D, return_index=True) 
        N = len(idx)
        
        # Normalisation
        R = r1D[-1]
        M = 4*np.pi * integrate(x=r1D[idx], y=r1D[idx]**2 * rho1D[idx])
        r   = r1D[idx] / (R)
        rho = rho1D[idx] / (M/R**3)
        P0  = surface_pressure / (M**2/R**4)
        
        # We assume that P0 is already normalised by G if R ~ 1 ...
        if not np.allclose(R, 1) :
            P0 /= G
            
    zeta = np.copy(r)
    
    return P0, N, M, R, r, zeta, rho, other_var

        
def init_2D() :
    """
    Init function for the angular domain.

    Parameters
    ----------
    None.

    Returns
    -------
    cth : array_like, shape (M, ), [GLOBAL VARIABLE]
        Angular coordinate (equivalent to cos(theta)).
    weights : array_like, shape (M, ), [GLOBAL VARIABLE]
        Angular weights for the Legendre quadrature.
    map_n : array_like, shape (N, M)
        Isopotential mapping 
        (given by r(phi_eff, theta) = r for now).
    """
    cth, weights = roots_legendre(M)
    map_n = np.tile(r, (M, 1)).T
    return cth, weights, map_n  

def init_phi_c() : 
    """
    Defines the functions used to compute the centrifugal potential
    and the rotation profile with the adequate arguments.

    Returns
    -------
    phi_c : function(r, cth, omega)
        Centrifugal potential
    w : function(r, cth, omega)
        Rotation profile

    """
    nb_args = PROFILE.__code__.co_argcount - len(PROFILE.__defaults__ or '')
    mask = np.array([0, 1]) < nb_args - 3
    
    # Creation of the centrifugal potential function
    args_phi = np.array([ALPHA, SCALE])[mask]
    phi_c = lambda r, cth, omega : PROFILE(r, cth, omega, *args_phi)
    
    # Creation of the rotation profile function
    args_w = np.hstack((np.atleast_1d(args_phi), (True,)))
    w = lambda r, cth, omega : PROFILE(r, cth, omega, *args_w)
    return phi_c, w


def find_pressure(rho, dphi_eff) :
    """
    Find the pressure evaluated on r thanks to the hydrostatic
    equilibrium.

    Parameters
    ----------
    rho : array_like, shape (N, )
        Density profile.
    dphi_eff : array_like, shape (N, )
        Effective potential derivative with respect to r.

    Returns
    -------
    P : array_like, shape (N, )
        Pressure profile.

    """
    dP = - rho * dphi_eff
    P  = interpolate_func(r, dP, der=-1, k=KSPL, prim_cond=(-1, P0))(r)
    return P


def find_rho_l(map_n, rho_n) : 
    """
    Find the density distribution harmonics from a given mapping 
    (map_n) which gives the lines of constant density (=rho_n).

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        Current mapping.
    rho_n : array_like, shape (N, )
        Current density on each equipotential.

    Returns
    -------
    rho_l : array_like, shape (N, L)
        Density distribution harmonics.

    """
    # Density interpolation on the mapping
    safety_constant = 1e-15
    all_k   = np.arange((M+1)//2)
    log_rho = np.log(rho_n + safety_constant)
    rho2D   = np.zeros((N, M))
    for k in all_k :
        inside =  r < map_n[-1, k]
        rho2D[inside, k] = interpolate_func(
            x=map_n[:, k], y=log_rho, k=KSPL
        )(r[inside])
        rho2D[inside, k] = np.exp(rho2D[inside, k]) - safety_constant
    rho2D[:,-1-all_k] = rho2D[:, all_k]
    
    # Corresponding harmonic decomposition
    rho_l = pl_project_2D(rho2D, L)
    
    return rho_l

def filling_ab(ab, ku, kl, l) : 
    """
    Fill the band storage matrix according to the Poisson's
    equation (written in terms of r^2). We optimize this
    operation using the scipy sparse matrix storage (the 
    [::-1] comes from the opposite storage convention 
    between Fortran and Scipy) and exploiting the fact that 
    most of the matrix stay the same when changing l.

    Parameters
    ----------
    ab : array_like, shape (ldmat, 2*N)
        Band storage matrix.
    ku : integer
        Number of terms in the upper matrix part.
    kl : integer
        Number of terms in the lower matrix part.
    l : integer
        Harmonic degree.

    Returns
    -------
    ab : array_like, shape (ldmat, 2*N)
        Filled band storage matrix.

    """    
    # Offset definition
    offset = ku + kl
    
    # The common filling part
    if l == 0 :
        ab[ku+1+(0-0):-1+(0-0):2, 0::2] =  Asp.data[::-1]
        ab[ku+1+(1-0):        :2, 0::2] = -Lsp.data[::-1]
        ab[ku+1+(1-1):-1+(1-1):2, 1::2] =  Dsp.data[::-1]
                
        # First boundary condition (l = 0)
        ab[offset, 0] = 6.0
        
    # The only l dependent part (~1/4 of the matrix)
    else : 
        ab[ku+1+(0-1):-1+(0-1):2, 1::2] = -l*(l+1) * Lsp.data[::-1]
        
        # First boundary condition (l != 0)
        ab[offset-0, 0] = 0.0
        ab[offset-1, 1] = 1.0
    
    # Boundary conditions
    ab[offset+1, 2*N-2] = 2*r[-1]**2
    ab[offset+0, 2*N-1] = l+1 
    return ab
    
    
def find_phi_eff(map_n, rho_n, phi_eff=None, lub_l=None) :
    """
    Determination of the effective potential from a given mapping
    (map_n, which gives the lines of constant density), and a given 
    rotation rate (omega_n). This potential is determined by solving
    the Poisson's equation on each degree of the harmonic decomposition
    (giving the gravitational potential harmonics which are also
    returned) and then adding the centrifugal potential.

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        Current mapping.
    rho_n : array_like, shape (N, )
        Current density on each equipotential.
    phi_eff : array_like, shape (N, ), optional
        If given, the current effective potential on each 
        equipotential. If not given, it will be calculated inside
        this fonction. The default is None.
    lub_l : list (size: Nl) of tuples (size: 2), optional
        Each element of the list contains contains the LU
        decomposition of Poisson's matrix (first tuple element)
        and the corresponding pivot indices (second tuple element)
        to solve Poisson's equation at a diven degree l. The 
        routine therefore fully exploit the invariance of Poisson's 
        matrix by computing those two elements once and for all.
        The default is None.

    Raises
    ------
    ValueError
        If the matrix inversion enconters a difficulty ...

    Returns
    -------
    phi_g_l : array_like, shape (N, L)
        Gravitation potential harmonics.
    dphi_g_l : array_like, shape (N, L)
        Gravitation potential harmonics derivative with respect to r^2.
    phi_eff : array_like, shape (N, )
        Effective potential on each equipotential.
    dphi_eff : array_like, shape (N, ), optional
        Effective potential derivative with respect to r^2.
    lub_l : list (size: Nl) of tuples (size: 2), optional
        Cf. parameters

    """    
    # Density distribution harmonics
    rho_l    = find_rho_l(map_n, rho_n)
    phi_g_l  = np.zeros((N, L))
    dphi_g_l = np.zeros((N, L))
    
    # Vector filling (vectorial)
    Nl = (L+1)//2
    bl = np.zeros((2*N, Nl))
    bl[1:-1:2, :] = 4*np.pi * Lsp @ (r[:,None]**2 * rho_l[:, ::2])
    bl[0     , 0] = 4*np.pi * rho_l[0, 0]     # Boundary condition
    
    # Band matrix storage
    kl = 2*KLAG
    ku = 2*KLAG
    ab = np.zeros((2*kl + ku + 1, 2*N))   
    
    if phi_eff is None :
        lub_l = []
        for l in range(0, L, 2) :
            # Matrix filling  
            ab = filling_ab(ab, ku, kl, l)
            
            # LU decomposition (LAPACK)
            lub_l.append(dgbtrf(ab, ku, kl)[:-1])
            
    # System solving (LAPACK)
    x = np.array([
        dgbtrs(lub_l[k][0], kl, ku, bl[:, k], lub_l[k][1])[0] for k in range(Nl)
    ]).T
        
    # Poisson's equation solution
    phi_g_l[: , ::2] = x[1::2]
    dphi_g_l[:, ::2] = x[0::2] * (2*r[:, None])  # <- The equation is solved on r^2
    
    if phi_eff is None :
        # First estimate of the effective potential and its derivative
        phi_eff  = pl_eval_2D( phi_g_l, 0.0)
        dphi_eff = pl_eval_2D(dphi_g_l, 0.0)        
        return phi_g_l, dphi_g_l, phi_eff, dphi_eff, lub_l
    
    # The effective potential is known to an additive constant 
    C = pl_eval_2D(phi_g_l[0], 0.0) - phi_eff[0]
    phi_eff += C
    return phi_g_l, dphi_g_l, phi_eff


def find_new_mapping(omega_n, phi_g_l, dphi_g_l, phi_eff) :
    """
    Find the new mapping by comparing the effective potential
    and the total potential (calculated from phi_g_l and omega_n).

    Parameters
    ----------
    omega_n : float
        Current rotation rate.
    phi_g_l : array_like, shape (N, L)
        Gravitation potential harmonics.
    dphi_g_l : array_like, shape (N, L)
        Gravitation potential derivative harmonics.
    phi_eff : array_like, shape (N, )
        Effective potential on each equipotential.

    Returns
    -------
    map_n_new : array_like, shape (N, M)
        Updated mapping.
    omega_n_new : float
        Updated rotation rate.

    """    
    # 2D gravitational potential (interior)
    up = np.arange((M+1)//2)
    phi2D_g_int  = pl_eval_2D( phi_g_l, cth[up])
    dphi2D_g_int = pl_eval_2D(dphi_g_l, cth[up])
    
    # 2D gravitational potential (exterior)
    l = np.arange(L)
    outside = 1.3        # Some guess
    r_ext = np.linspace(1.0, outside, 101)[1:]
    phi_g_l_ext  = phi_g_l[-1] * (r_ext[:, None])**-(l+1)
    dphi_g_l_ext = -(l+1) * phi_g_l_ext / r_ext[:, None]
    phi2D_g_ext  = pl_eval_2D( phi_g_l_ext, cth[up])
    dphi2D_g_ext = pl_eval_2D(dphi_g_l_ext, cth[up])
    
    # 2D gravitational potential
    r_tot = np.hstack((r, r_ext))
    phi2D_g  = np.vstack(( phi2D_g_int,  phi2D_g_ext))
    dphi2D_g = np.vstack((dphi2D_g_int, dphi2D_g_ext))
        
    # Find a new value for ROT
    valid_r = r_tot > 0.5
    phi1D_c = eval_phi_c(r_tot[valid_r], 0.0, omega_n)[0] / r_tot[valid_r] ** 3
    phi1D  =  phi2D_g[valid_r, (M-1)//2] + phi1D_c
    r_est = interpolate_func(x=phi1D, y=r_tot[valid_r], k=KSPL)(phi_eff[-1])
    omega_n_new = omega_n * r_est**(-1.5)
    
    # Centrifugal potential
    phi2D_c, dphi2D_c = np.moveaxis(np.array([
        eval_phi_c(r_tot , ck, omega_n_new) for ck in cth[up]
    ]), (0, 1, 2), (2, 0, 1))
    
    # Total potential
    phi2D  =  phi2D_g +  phi2D_c
    dphi2D = dphi2D_g + dphi2D_c
    
    # Finding the valid interpolation domain
    phi_valid = np.ones_like(phi2D, dtype='bool')
    for k, dpk in enumerate(dphi2D.T) :
        if np.any(dpk < 0.0) :
            idx_max = np.min(np.argwhere((dpk < 0.0)&(r_tot > 0.0)))
            phi_valid[:, k] = np.arange(len(r_tot)) < idx_max
    
    # Central domain
    lim = 5e-2
    center = np.max(np.argwhere(r_tot < lim)) + 1
    r_cnt = np.linspace(0.0, 1.0, 5*center) ** 2 * lim
    phi2D_g_cnt = np.array([CubicHermiteSpline(
        x=r, y=phi2D_g_int[:, k], dydx=dphi2D_g_int[:, k]
    )(r_cnt) for k in up]).T
    phi2D_c_cnt = np.array([
        eval_phi_c(r_cnt , ck, omega_n_new)[0] for ck in cth[up]
    ]).T
    phi2D_cnt = phi2D_g_cnt + phi2D_c_cnt
    
    # Estimate at target values
    map_est = np.vstack((
        np.zeros_like(up), 
        np.array([
            interpolate_func(x=pk, y=r_cnt, k=KSPL)(phi_eff[1:center]) 
            for pk in phi2D_cnt.T
        ]).T,
        np.array([
            interpolate_func(x=pk[valid_k], y=r_tot[valid_k], k=KSPL)(phi_eff[center:]) 
            for pk, valid_k in zip(phi2D.T, phi_valid.T)
        ]).T
    ))
            
    # New mapping
    map_n_new = np.hstack((map_est, np.flip(map_est, axis=1)[:, 1:]))
        
    return map_n_new, omega_n_new


def Virial_theorem(map_n, rho_n, omega_n, phi_eff, P, verbose=False) : 
    """
    Compute the Virial equation and gives the resukt as a diagnostic
    for how well the hydrostatic equilibrium is satisfied (the closer
    to zero, the better).
    
    Parameters
    ----------
    map_n : array_like, shape (N, M)
        Mapping
    rho_n : array_like, shape (N, )
        Density on each equipotential.
    omega_n : float
        Rotation rate.
    phi_eff : array_like, shape (N, )
        Effective potential on each equipotential.
    P : array_like, shape (N, )
        Pressure on each equipotential.
    verbose : bool
        Whether to print the individual energy values or not.
        The default is None.

    Returns
    -------
    virial : float
        Value of the normalised Virial equation.

    """    
    # Potential energy
    volumic_potential_energy = lambda rk, ck, D : (  
       rho_n[D] * (phi_eff[D]-eval_phi_c(rk[D], ck, omega_n)[0])
    )
    potential_energy = integrate2D(map_n, volumic_potential_energy, k=KSPL)
    
    # Kinetic energy
    volumic_kinetic_energy = lambda rk, ck, D : (  
       0.5 * rho_n[D] * (1-ck**2) * rk[D]**2 * eval_w(rk[D], ck, omega_n)**2
    )
    kinetic_energy = integrate2D(map_n, volumic_kinetic_energy, k=KSPL)
    
    # Internal energy
    internal_energy = integrate2D(map_n, P, k=KSPL)
    
    # Compute the virial equation
    if verbose :
        print(f"Kinetic energy  : {kinetic_energy:+7.5f}")
        print(f"Internal energy : {internal_energy:+7.5f}")
        print(f"Potential energy: {potential_energy:+7.5f}")
    virial = ( 
          (2*kinetic_energy + 0.5*potential_energy + 3*internal_energy)
        / (2*kinetic_energy - 0.5*potential_energy + 3*internal_energy)
    )
    return virial


def find_gravitational_moments(map_n, rho_n, max_degree=14) :
    """
    Find the gravitational moments up to max_degree.

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        Isopotential mapping.
    rho_n : array_like, shape (N, )
        Density profile (the same in each direction).
    max_degree : int, optional
        Maximum degree for the gravitational moments. The default is 14.
        
    Returns
    -------
    None.

    """
    print(
        "\n+-----------------------+",
        "\n| Gravitational moments |", 
        "\n+-----------------------+\n"
    )
    for l in range(0, max_degree+1, 2):
        m_l = integrate2D(
            map_n, rho_n[:, None] * map_n ** l * eval_legendre(l, cth), k=KSPL
        )
        print("Moment n°{:2d} : {:+.10e}".format(l, m_l))
        

def write_model(fname, map_n, *args) : 
    """
    Saves the deformed model in the file named fname. The resulting 
    table has dimension (N, M+N_args+N_var) where the last N_var columns
    contains the additional variables given by the user (the lattest
    are left unchanged during the whole deformation). The dimensions N & M,
    as well as the global paramaters mass, radius, ROT, G
    are written on the first line.

    Parameters
    ----------
    fname : string
        File name
    map_n : array_like, shape (N, M)
        level surfaces mapping.
    args : TUPLE with N_args elements
        Variables to be saved in addition to map_n.

    """
    np.savetxt(
        'Models/'+fname, np.hstack((map_n, np.vstack(args + (*VAR,)).T)), 
        header=f"{N} {M} {mass} {radius} {ROT} {G}", 
        comments=''
    )
    
#%% Main cell

if __name__ == '__main__' :
    
    start = time.perf_counter()
    
    # Definition of global parameters
    MOD_1D, ROT, FULL, PROFILE, ALPHA, SCALE, EPS, KSPL, \
        KLAG, L, M, RES, SAVE = set_params() 
    G = 6.67384e-8     # <- value of the gravitational constant
        
    # Definition of the 1D-model
    P0, N, mass, radius, r, zeta, rho_n, VAR = init_1D()  
    
    # Angular domain preparation
    cth, weights, map_n      = init_2D()
    
    # Centrifugal potential definition
    eval_phi_c, eval_w = init_phi_c()
    
    # Find the lagrange matrix
    lag_mat = lagrange_matrix_P(r**2, order=KLAG)
    
    # Define sparse matrices 
    Lsp = sps.dia_matrix(lag_mat[..., 0])
    Dsp = sps.dia_matrix(lag_mat[..., 1])
    Asp = sps.dia_matrix(
        4 * lag_mat[..., 1] * r**4 - 2 * lag_mat[..., 0] * r**2
    )
    
    # Initialisation for the effective potential
    phi_g_l, dphi_g_l, phi_eff, dphi_eff, lub_l = find_phi_eff(map_n, rho_n)
    
    # Find pressure
    P = find_pressure(rho_n, dphi_eff)
    
    # Iterative centrifugal deformation
    surfaces = [map_n[-1]]
    r_pol = [0.0, find_r_pol(map_n, L)]
    n = 1
    print(
        "\n+---------------------+",
        "\n| Deformation started |", 
        "\n+---------------------+\n"
    )
    
    # SAVE
    phi_g_l_rad  = [phi_g_l]
    dphi_g_l_rad = [dphi_g_l]
    phi_eff_rad  = [np.copy(phi_eff)]
    map_n_rad    = [map_n]
    omega_n_rad  = [0.0]
    
    
    while abs(r_pol[-1] - r_pol[-2]) > EPS :
        
        # Current rotation rate
        omega_n = min(ROT, (n/FULL) * ROT)
        
        # Effective potential computation
        phi_g_l, dphi_g_l, phi_eff = find_phi_eff(map_n, rho_n, phi_eff, lub_l)
    
        # SAVE
        phi_g_l_rad.append(phi_g_l)
        dphi_g_l_rad.append(dphi_g_l)
        phi_eff_rad.append(np.copy(phi_eff))

        # Update the mapping
        map_n, omega_n = find_new_mapping(omega_n, phi_g_l, dphi_g_l, phi_eff)

        # SAVE
        map_n_rad.append(map_n)
        omega_n_rad.append(omega_n)
        
        # Renormalisation
        r_corr    = find_r_eq(map_n, L)
        m_corr    = integrate2D(map_n, rho_n, k=KSPL)
        radius   *= r_corr
        mass     *= m_corr
        map_n    /=             r_corr
        rho_n    /= m_corr    / r_corr**3
        phi_eff  /= m_corr    / r_corr
        dphi_eff /= m_corr    / r_corr**2
        P        /= m_corr**2 / r_corr**4
        
        # Update the surface and polar radius
        surfaces.append(map_n[-1])
        r_pol.append(find_r_pol(map_n, L))
        
        # Iteration count
        DEC = int(-np.log10(EPS))
        print(f"Iteration n°{n:02d}, R_pol = {r_pol[-1].round(DEC)}")
        n += 1
    
    # Deformation summary
    finish = time.perf_counter()
    print(
        "\n+------------------+",
        "\n| Deformation done |", 
        "\n+------------------+\n"
    )
    print(f'Time taken: {round(finish-start, 2)} secs')  
    estimated_prec = np.max(np.abs(phi_g_l[:, -1]/phi_g_l[:, 0]))
    print(f"Estimated error on Poisson's equation: {round(estimated_prec, 16)}")   
    virial = Virial_theorem(map_n, rho_n, omega_n, phi_eff, P, verbose=True)
    print(f"Virial theorem verified at {round(virial, 16)}")   
    
    # Plot mapping
    plot_f_map(map_n, rho_n, phi_eff, L, show_surfaces=True)        
    
    # Gravitational moments
    find_gravitational_moments(map_n, rho_n)
    
    # # Some diagnostics
    # from utils import phi_g_harmonics, check_interpolation
    # step = n-1
    # phi_g_harmonics(r, phi_g_l_rad[step], r_pol[step])
    # check_interpolation(r, map_n_rad[step], rho_n)
        
    # # Model scaling
    # map_n    *=               radius
    # rho_n    *=     mass    / radius**3
    # phi_eff  *= G * mass    / radius   
    # dphi_eff *= G * mass    / radius**2
    # P        *= G * mass**2 / radius**4
    
    # # Model writing
    # write_model(SAVE, map_n, r, P, rho_n, phi_eff)
    
    
        