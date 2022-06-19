import numpy as np
from scipy.integrate import simpson

from .units import *
from .halos import rho_spike
from .basic import E_orb, L_orb
from .eccentricity import *
from . import HaloFeedback

# ========== Energy functions

def csection_accretion(Deltau, m2, full = False):
    """ The cross section of black hole body m2 [M_sun] accreting CDM at velocity u [m/s]."""
    
    if full:
        uc2 = (Deltau/c)**2
        term = 1/4 *(8 *(1 -uc2))**3 / (1 -4 *uc2 +np.sqrt(1 +8 *uc2)) / (3 -np.sqrt(1 +8 *uc2))**2
        
        return (np.pi *G**2 *(m2 *Mo)**2 /uc2 /c**4 *term).astype(float)
    else:
        return 16 *np.pi *G**2 *(m2 *Mo)**2 /c**2 *(1/Deltau**2 +1/c**2)

def ksiRate(r, u, spike):
    psi = spike.psi(r) # [km2/s2]
    c3 = c/1000 # [km/s]
    
    v = np.sqrt(2 *(psi -spike.eps_grid))
    mask = (spike.eps_grid < psi)
    f_v = 4 *np.pi *spike.f_eps/spike.rho(r) *v**2
    
    p = u+v; d = np.abs(u -v)
    terms = (p -d) + (u**2 /c3**2) *(p**3 -d**3)/3/u**2
    integrand = f_v *terms/2/v
    
    return simpson(integrand[mask], v[mask])/(1 +u**2 /c3**2)

ksiRate = np.vectorize(ksiRate)

def ksi_Facc(r, u, spike):
    psi = spike.psi(r) # [km2/s2]
    c3 = c/1000 # [km/s]
    
    v = np.sqrt(2 *(psi -spike.eps_grid))
    mask = (spike.eps_grid < psi)
    f_v = 4 *np.pi *spike.f_eps/spike.rho(r) *v**2
    
    p = u+v; d = np.abs(u -v)
    terms = (u**2 -v**2) *(p -d) +(p**3 -d**3)/3 *(1 +(u**2 -v**2)/c3**2) +(p**5 -d**5)/(5*c3**2)
    integrand = f_v *terms/4/v/u**2
    
    return simpson(integrand[mask], v[mask])/(1 +u**2 /c3**2)

ksi_Facc = np.vectorize(ksi_Facc)

def F_Acc(r: float, u: float, spike: HaloFeedback.DistributionFunction, isStaticCDM = False, phaseSpace = True) -> float:
    """ Calculates the dissipative force of dynamical friction of a binary-spike system at a position r from
    the center and with orbital velocity u. The force is connected to the energy loss as dEdt_DF = F_DF *u.
    
    * spike is a HaloFeedback spike with all information about the components and dark matter distribution.
    * r is the distance [pc] from the center of the spike.
    * u is the orbital velocity [m/s] of the secondary component interacting with the spike.
    * phaseSpace controls wether the kinematics of the spike will be taken into consideration in the friction.
    """
    m1 = spike.m1; m2 = spike.m2
    
    if isStaticCDM:
        rho = rho_spike(r, spike.gamma, spike.rho_sp, m1) # [kg/m3]
    else:
        rho = spike.rho(r) *Mo/pc**3 # [kg/m3]
    
    ksi = ksi_Facc(r, u/1000, spike) and phaseSpace or 1
    
    F = u**2 *rho *csection_accretion(u, m2) *ksi
    
    return F

def dm2dt_Acc(r: float, u: float, spike: HaloFeedback.DistributionFunction, isStaticCDM = False, phaseSpace = False) -> float:
    """
    """
    m1 = spike.m1; m2 = spike.m2
    
    if isStaticCDM:
        rho = rho_spike(r, spike.gamma, spike.rho_sp, m1) # [kg/m3]
    else:
        rho = spike.rho(r) *Mo/pc**3 # [kg/m3]
    
    ksi = ksiRate(r, u/1000, spike) if phaseSpace else 1
    dmdt = u *rho *csection_accretion(u, m2) *ksi
    
    return dmdt # [kg/s]

def averageAccLossRates(spike: HaloFeedback.DistributionFunction, a: float, e: float, isStaticCDM = False, phaseSpace = False) -> tuple:
    """ Calculates the time-averaged energy and angular momentum loss rates experienced by the components of the
    spike-halo system due to dynamical friction over a single orbit.
    
    * spike is a HaloFeedback distribution function that describes the system.
    * a is the semi-major axis [pc] of the orbit.
    * e is the eccentricity of the orbit.
    * isStaticCDM controls if a static CDM spike will be assumed based on the spike's parameters.
    * phaseSpace controls if the static CDM spike will have constant or moving particles.
    """
    m = spike.m() # [M_sun] The total mass.
    
    if e == 0: # Bypass the force averaging for circular orbits.
        u = getOrbitalVelocity(a, 0, 0, m) # [m/s]
        dEdt = F_Acc(a, u, spike, isStaticCDM = isStaticCDM, phaseSpace = phaseSpace) *u
        dLdt = dEdt /np.sqrt(G *m *Mo/ (a *pc)**3)
        dm2dt = dm2dt_Acc(a, u, spike, isStaticCDM = isStaticCDM, phaseSpace = phaseSpace) /Mo
        
        return dEdt, dLdt, dm2dt # [M_sun/s]
    
    # Generate the force around the orbit.
    N_grid = 2 *getGridSizeForEccentricity(e)
    theta = np.linspace(0, np.pi, N_grid) # [rad]
    
    r = getSeparation(a, e, theta) # [pc]
    u = getOrbitalVelocity(a, e, theta, m) # [m/s]
    F = np.vectorize(F_Acc)(r, u, spike, isStaticCDM = isStaticCDM, phaseSpace = phaseSpace) # [kg m/s2]
    dmdt = dm2dt_Acc(a, u, spike, isStaticCDM = isStaticCDM, phaseSpace = phaseSpace) # [kg/s]
    
    # Integrate forces and add weights.
    intE = simpson(F *u / (1 +e *np.cos(theta))**2, theta/np.pi)
    intL = simpson(F /u / (1 +e *np.cos(theta))**2, theta/np.pi)
    intM = simpson(dmdt/ (1 +e *np.cos(theta))**2, theta/np.pi)
    
    dEdt = (1 -e**2)**(3/2) *intE
    dLdt = (1 -e**2)**(3/2) *intL *np.sqrt(G *(m *Mo) *(a *pc) *(1 -e**2))
    dm2dt = (1 -e**2)**(3/2) *intM /Mo # [M_sun/s]
    
    return dEdt, dLdt, dm2dt

# def getOrbitUpdateAcc(dm2dt: float, a: float, e: float, m1: float, m2: float) -> tuple:
#     dadt = a *dm2dt/m2 # [pc/s]
    
#     if e > 0:
#         m = m1 +m2; mu = m1 *m2/m # [Mo]
#         G = -(2/m +3 *mu/m**2) *dm2dt
        
#         dedt = -(1 -e**2)/2/e *(G) # [1/s]
#     else:
#         dedt = 0
    
#     return dadt, dedt

# getOrbitUpdateAcc = np.vectorize(getOrbitUpdateAcc)