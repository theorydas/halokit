import numpy as np
from scipy.integrate import simpson

from .units import *
from .halos import rho_spike
from .basic import E_orb, L_orb
from .eccentricity import *
from . import HaloFeedback

# ========== Energy functions

def csection_accretion(u, m2):
    """ The cross section of black hole body m2 [M_sun] accreting CDM at velocity u [m/s]."""
    uc2 = (u/c)**2
    
    term = 1/4 *(8 *(1 -uc2))**3 / (1 -4 *uc2 +np.sqrt(1 +8 *uc2)) / (3 -np.sqrt(1 +8 *uc2))**2
    
    return (np.pi *G**2 *(m2 *Mo)**2 /uc2 /c**4 *term).astype(float)

def F_Acc(r: float, u: float, spike: HaloFeedback.DistributionFunction, isStaticCDM = False, useRadius = False) -> float:
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
    
    if not useRadius:
        F = u**2 *rho *csection_accretion(u, m2)
    else:
        F = u**2 *rho *(np.pi *spike.b_min(r, u) *pc)**2
    
    return F

def averageAccLossRates(spike: HaloFeedback.DistributionFunction, a: float, e: float, isStaticCDM = False, useRadius = False) -> tuple:
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
        dEdt = F_Acc(a, u, spike, isStaticCDM = isStaticCDM, useRadius = useRadius) *u
        dLdt = dEdt /np.sqrt(G *m *Mo/ (a *pc)**3)
        dm2dt = dEdt/u**2 /Mo
        
        return dEdt, dLdt, dm2dt # [M_sun/s]
    
    # Generate the force around the orbit.
    N_grid = 2 *getGridSizeForEccentricity(e)
    theta = np.linspace(0, np.pi, N_grid) # [rad]
    
    r = getSeparation(a, e, theta) # [pc]
    u = getOrbitalVelocity(a, e, theta, m) # [m/s]
    F = np.vectorize(F_Acc)(r, u, spike, isStaticCDM = isStaticCDM, useRadius = useRadius) # [kg m/s2]
    
    # Integrate forces and add weights.
    int1 = simpson(F *u / (1 +e *np.cos(theta))**2, theta/np.pi)
    int2 = simpson(F /u / (1 +e *np.cos(theta))**2, theta/np.pi)
    
    dEdt = (1 -e**2)**(3/2) *int1
    dLdt = (1 -e**2)**(3/2) *int2 *np.sqrt(G *(m *Mo) *(a *pc) *(1 -e**2))
    dm2dt = (1 -e**2)**(3/2) *int2 /Mo # [M_sun/s]
    
    return dEdt, dLdt, dm2dt

def averageAccLossRates2(spike: HaloFeedback.DistributionFunction, a: float, e: float, isStaticCDM = False, dmdt = 1) -> tuple:
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
        dEdt = dmdt *u**2
        dLdt = dEdt /np.sqrt(G *m *Mo/ (a *pc)**3)
        dm2dt = dEdt/u**2 /Mo
        
        return dEdt, dLdt, dm2dt # [M_sun/s]
    
    # Generate the force around the orbit.
    N_grid = 2 *getGridSizeForEccentricity(e)
    theta = np.linspace(0, np.pi, N_grid) # [rad]
    
    r = getSeparation(a, e, theta) # [pc]
    u = getOrbitalVelocity(a, e, theta, m) # [m/s]
    F = np.vectorize(F_Acc)(r, u, spike, isStaticCDM = isStaticCDM) # [kg m/s2]
    
    # Integrate forces and add weights.
    int1 = simpson(F *u / (1 +e *np.cos(theta))**2, theta/np.pi)
    int2 = simpson(F /u / (1 +e *np.cos(theta))**2, theta/np.pi)
    
    dEdt = (1 -e**2)**(3/2) *int1
    dLdt = (1 -e**2)**(3/2) *int2 *np.sqrt(G *(m *Mo) *(a *pc) *(1 -e**2))
    dm2dt = (1 -e**2)**(3/2) *int2 /Mo # [M_sun/s]
    
    return dEdt, dLdt, dm2dt