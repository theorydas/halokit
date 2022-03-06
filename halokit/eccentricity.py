import numpy as np
from scipy.integrate import simpson

from .units import *
from .halos import getKsiCDM, rho_spike
from . import HaloFeedback

def getGridSizeForEccentricity(e: float) -> float:
    """ Returns an empirical grid size for integration of eliptical orbits roughly based on
    the eccentricity number to improve speed when convergence is easier with fewer points.
    """
    if e < 0.01:
        N_grid = 3
    elif e < 0.3:
        N_grid = 20
    elif e < 0.6:
        N_grid = 40
    else:
        N_grid = 100
    
    return N_grid

def getEccentricDistance(a: float, e: float, phi: float) -> float:
    """ Calculates the real distance [a units] from the center of an elliptic orbit with eccentricity e and semimajor
    axis a at the true anomaly phi.
    
    * a is the semi-major axis of the orbit.
    * e is the eccentricity.
    * phi is the true anomaly [rad].
    """
    if e < 0 or e >= 1: raise ValueError("The eccentricity must be within the range [0, 1).")
    
    p = a *(1 -e**2)
    r = p/(1 +e *np.cos(phi))
    
    return r

def getEccentricVelocity(a: float, e: float, phi: float, m: float) -> float:
    """ Calculates the orbital velocity [m/s] of an elliptic orbit with eccentricity e and semimajor
    axis a at the true anomaly phi and components with total mass m.
    
    * a is the semi-major axis [pc] of the orbit.
    * e is the eccentricity.
    * phi is the true anomaly [rad].
    * m is the total mass [M_sun] of the components.
    """
    if e < 0 or e >= 1: raise ValueError("The eccentricity must be within the range [0, 1).")
    
    r = getEccentricDistance(a, e, phi) # [pc]
    u2 = G *m *Mo *(2*a -r)/a/r /pc # [m2/s2] Scrambled terms to avoid numerical truncation.
    
    return np.sqrt(u2) # [m/s]

# ========== Energy functions

def dEdt_GW(m1, m2, a, e):
    """ Calculate the orbit averaged energy loss due to gravitational wave emission of an
    orbit with eccentricity e and semi major axis a per Eq. 15 of 2112.09586v1.
    
    * m1, m2 in M_sun.
    * a the semi major axis in pc.
    """
    m = m1 +m2 # [M_sun] The total mass of the binary.
    mu = m1*m2/m # The reduced mass of the binary.
    
    dEdt = 32/5 *mu**2 *m**3/(a *pc)**5 *G**4 / c**5 *Mo**5
    dEdt *= (1 +73/24 *e**2 +37/96 *e**4) *(1 -e**2)**(-7/2) # Weight because of non-zero eccentricity.
    
    return dEdt

def dLdt_GW(m1, m2, a, e):
    """ Calculate the orbit averaged angular momentum loss due to gravitational wave emission of an
    orbit with eccentricity e and semi major axis a per Eq. 16 of 2112.09586v1.
    
    * m1, m2 in M_sun.
    * a the semi major axis in pc.
    """
    m = m1 +m2 # The total mass of the binary.
    mu = m1*m2/m # The reduced mass of the binary.
    
    dLdt = 32/5 *(mu *Mo)**2 *(m *Mo)**(5/2)/(a *pc)**(7/2) *G**(7/2) / c**5
    dLdt *= (1 +7/8*e**2) /(1 -e**2)**2 # Weight because of non-zero eccentricity.
    
    return dLdt

def F_DF(r: float, u: float, spike: HaloFeedback.DistributionFunction, isStaticCDM = False, phaseSpace: bool = True) -> float:
    """ Calculates the dissipative force of dynamical friction of a binary-spike system at a position r from
    the center and with orbital velocity u. The force is connected to the energy loss as dEdt_DF = F_DF *u.
    
    * spike is a HaloFeedback spike with all information about the components and dark matter distribution.
    * r is the distance [pc] from the center of the spike.
    * u is the orbital velocity [m/s] of the secondary component interacting with the spike.
    * phaseSpace controls wether the kinematics of the spike will be taken into consideration in the friction.
    """
    m1 = spike.M_BH; m2 = spike.M_NS
    
    Lambda = np.sqrt(m1/m2)
    
    if isStaticCDM:
        # The maximum velocity of allowed DM particles at position r.
        umax = np.sqrt(2 *G *m1 *Mo/r/pc) # [m/s]
        
        # The fraction of the density with DM particles moving u < uorb of the companion.
        ksi = getKsiCDM(m2/m1, spike.gamma, u/umax) if phaseSpace else 1 #getKsiCDM(m2/m1, spike.gamma)
        rho = rho_spike(r, spike.gamma, spike.rho_sp, m1) *ksi
    else:
        rho = spike.rho(r, v_cut = u/1000) *Mo/pc**3 # [kg/m3]
    
    F = 4 *np.pi *(G *m2 *Mo)**2 *np.log(Lambda) /u**2 *rho
    
    return F

def averageLossRates(spike: HaloFeedback.DistributionFunction, a: float, e: float, isStaticCDM = False, phaseSpace: bool = True, N_grid = 40) -> tuple:
    m1 = spike.M_BH; m2 = spike.M_NS
    m = m1 +m2 # [M_sun] The total mass.
    
    # Generate the force around the orbit.
    N_grid = getGridSizeForEccentricity(e)
    
    phi = np.linspace(0, np.pi, N_grid) # [rad]
    r = getEccentricDistance(a, e, phi) # [pc]
    u = getEccentricVelocity(a, e, phi, m) # [m/s]
    F = np.vectorize(F_DF)(r, u, spike, isStaticCDM = isStaticCDM, phaseSpace = phaseSpace) # [kg m/s2]
    
    # Integrate forces and add weights.
    int1 = 2 *simpson(F *u / (1 +e *np.cos(phi))**2, phi)
    int2 = 2 *simpson(F /u / (1 +e *np.cos(phi))**2, phi)
    
    dEdt = (1 -e**2)**(3/2) /2/np.pi *int1
    dLdt = (1 -e**2)**(3/2) /2/np.pi *int2 *np.sqrt(G *(m *Mo) *(a *pc) *(1 -e**2))
    
    return dEdt, dLdt