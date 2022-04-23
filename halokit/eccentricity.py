import numpy as np
from scipy.integrate import simpson

from .units import *
from .halos import getKsiCDM, rho_spike
from .basic import E_orb, L_orb
from . import HaloFeedback

def getGridSizeForEccentricity(e: float) -> float:
    """ Returns an empirically calcultaed grid size for integration of eliptical orbits roughly based on
    the eccentricity number to improve speed when convergence is easier on lower eccentricities.
    """
    if e == 0:
        N_grid = 1
    elif e < 0.01:
        N_grid = 15
    elif e < 0.3:
        N_grid = 50
    elif e < 0.6:
        N_grid = 90
    else:
        N_grid = 100
    
    return N_grid

def getSeparation(a: float, e: float, theta: float) -> float:
    """ Calculates the real distance [a units] from the center of an elliptic orbit with eccentricity e and semimajor
    axis a at the true anomaly theta.
    
    * a is the semi-major axis of the orbit.
    * e is the eccentricity.
    * theta is the true anomaly [rad].
    """
    if e < 0 or e >= 1: raise ValueError("The orbit eccentricity must be within the range [0, 1).")
    
    p = a *(1 -e**2) # [a units]
    
    return p/(1 +e *np.cos(theta)) # [a units]

def getOrbitalVelocity(a: float, e: float, theta: float, m: float) -> float:
    """ Calculates the orbital velocity [m/s] of an elliptic orbit with eccentricity e and semimajor
    axis a at the true anomaly theta and components with total mass m.
    
    * a is the semi-major axis [pc] of the orbit.
    * e is the eccentricity.
    * theta is the true anomaly [rad].
    * m is the total mass [M_sun] of the components.
    """
    if e < 0 or e >= 1: raise ValueError("The orbit eccentricity must be within the range [0, 1).")
    
    r = getSeparation(a, e, theta) # [pc]
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
    mu = m1*m2/m # [M_sun] The reduced mass of the binary.
    
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
    mu = m1*m2/m # [M_sun] The reduced mass of the binary.
    
    dLdt = 32/5 *(mu *Mo)**2 *(m *Mo)**(5/2)/(a *pc)**(7/2) *G**(7/2) / c**5
    dLdt *= (1 +7/8*e**2) /(1 -e**2)**2 # Weight because of non-zero eccentricity.
    
    return dLdt

def F_DF(r: float, u: float, spike: HaloFeedback.DistributionFunction, isStaticCDM = False, phaseSpace: bool = True, Lambda = True) -> float:
    """ Calculates the dissipative force of dynamical friction of a binary-spike system at a position r from
    the center and with orbital velocity u. The force is connected to the energy loss as dEdt_DF = F_DF *u.
    
    * spike is a HaloFeedback spike with all information about the components and dark matter distribution.
    * r is the distance [pc] from the center of the spike.
    * u is the orbital velocity [m/s] of the secondary component interacting with the spike.
    * phaseSpace controls wether the kinematics of the spike will be taken into consideration in the friction.
    """
    m1 = spike.m1; m2 = spike.m2
    
    if Lambda == True:
        Lambda_ = spike.Lambda(r, u/1000)#(r *pc) *u**2 / (G *np.sqrt(m1 *m2 *Mo**2))
        # print(1 -spike.Lambda(r, u/1000)/Lambda_)
    elif Lambda == False:
        Lambda_ = np.sqrt(m1/m2)
    else:
        Lambda_ = Lambda
    
    if isStaticCDM:
        # The maximum velocity of allowed DM particles at position r.
        umax = np.sqrt(2 *G *m1 *Mo/r/pc) # [m/s]
        
        # The fraction of the density with DM particles moving u < uorb of the companion.
        ksi = getKsiCDM(m2/m1, spike.gamma, u/umax) if phaseSpace else 1
        rho = rho_spike(r, spike.gamma, spike.rho_sp, m1) *ksi # [kg/m3]
    else:
        if phaseSpace:
            rho = spike.rho(r, v_cut = u/1000) *Mo/pc**3 # [kg/m3]
        else:
            rho = spike.rho(r) *Mo/pc**3 # [kg/m3]
    
    F = 4 *np.pi *(G *m2 *Mo)**2 *np.log(Lambda_) /u**2 *rho
    
    return F

def averageDFLossRates(spike: HaloFeedback.DistributionFunction, a: float, e: float, isStaticCDM = False, phaseSpace: bool = True, Lambda = True) -> tuple:
    """ Calculates the time-averaged energy and angular momentum loss rates experienced by the components of the
    spike-halo system due to dynamical friction over a single orbit.
    
    * spike is a HaloFeedback distribution function that describes the system.
    * a is the semi-major axis [pc] of the orbit.
    * e is the eccentricity of the orbit.
    * isStaticCDM controls if a static CDM spike will be assumed based on the spike's parameters.
    * phaseSpace controls if the static CDM spike will have constant or moving particles.
    """
    m1 = spike.m1; m2 = spike.m2
    m = m1 +m2 # [M_sun] The total mass.
    
    if e == 0: # Bypass the force averaging for circular orbits.
        u = getOrbitalVelocity(a, 0, 0, m) # [m/s]
        dEdt = F_DF(a, u, spike, isStaticCDM = isStaticCDM, phaseSpace = phaseSpace, Lambda = Lambda) *u
        dLdt = dEdt /np.sqrt(G *m *Mo/ (a *pc)**3)
        
        return dEdt, dLdt
    
    # Generate the force around the orbit.
    N_grid = 2 *getGridSizeForEccentricity(e)
    theta = np.linspace(0, np.pi, N_grid) # [rad]
    
    r = getSeparation(a, e, theta) # [pc]
    u = getOrbitalVelocity(a, e, theta, m) # [m/s]
    F = np.vectorize(F_DF)(r, u, spike, isStaticCDM = isStaticCDM, phaseSpace = phaseSpace, Lambda = Lambda) # [kg m/s2]
    
    # Integrate forces and add weights.
    int1 = simpson(F *u / (1 +e *np.cos(theta))**2, theta/np.pi)
    int2 = simpson(F /u / (1 +e *np.cos(theta))**2, theta/np.pi)
    
    dEdt = (1 -e**2)**(3/2) *int1
    dLdt = (1 -e**2)**(3/2) *int2 *np.sqrt(G *(m *Mo) *(a *pc) *(1 -e**2))
    
    return dEdt, dLdt

def getOrbitUpdate(dEdt: float, dLdt: float, a: float, e: float, m1: float, m2: float) -> tuple:
    dadt = dEdt *2 *(a *pc)**2 / (G *m1 *m2 *Mo**2) /pc # [pc/s]
    
    if e > 0:
        dedt = -(1 -e**2)/2/e *(dEdt/E_orb(a, m1, m2) +2 *dLdt/L_orb(a, e, m1, m2)) # [1/s]
    else:
        dedt = 0
    
    return dadt, dedt

getOrbitUpdate = np.vectorize(getOrbitUpdate)