from halokit.units import *

import numpy as np

def E_orb(a: float, m1: float, m2: float) -> float:
  """ Calculates the orbital energy [kg*m2/s2] of a binary system.

  * a is the semi-major axis [pc] of the two components's orbit or the separation for circular orbits.
  * m1, m2 are the masses [M_sun] of the components.
  """

  return - G *m1 *m2 *Mo**2 /(2 *a *pc)

def L_orb(a: float, e: float, m1: float, m2: float) -> float:
    """ Calculates the orbital angular momentum [kg*m2/s] of a binary system.
    
    * m1, m2 are the masses [M_sun] of the components.
    * a is the semi-major axis [pc] of the two components's orbit or the separation for circular orbits.
    * e is the eccentricity of the orbit.
    """
    if e < 0 or e >= 1: raise ValueError("The eccentricity must be within the range [0, 1).")
    
    m = m1 +m2 # [M_sun]
    p = a *(1 -e**2) *pc # [m]
    
    L = m1 *m2 *np.sqrt(G *p/(m *Mo)) *Mo**2 # [kg*m2/s]
    
    return L # [kg*m2/s]

L_orb = np.vectorize(L_orb)

def getRisco(m: float) -> float:
  """ Calculates the radius [pc] of the Innermost Stable Circular Orbit
  for a massive object of mass m [M_sun].
  """
  
  return 6 *G *m *Mo/c**2 /pc # [pc] # = 3 Rs

def getFisco(m1: float, m2: float) -> float:
  """ Calculates the orbital frequency [Hz] of the Innermost Stable Circular Orbit
  for a massive object of mass m1 [M_sun] in a binary system with another object m2 [M_sun].
  """

  return 1/getPeriodFromDistance(getRisco(m1), m1 +m2)

def getOrbitalFrequency(a: float, M_tot: float, mu: float = 0):
  """ Calculates the Kepplerian orbital frequency [Hz] of the binary system with
  total mass [M_sun] M_tot at a semi-major axis a [pc]. If mu is provided, it will calculate
  1.5 PN-corrected orbital frequency.
  """
  
  return 1

def getPeriodFromDistance(a: float, M_tot: float) -> float:
  """ Calculates the Kepplerian period [s] of binary system with
  total mass [M_sun] M_tot at a semi-major axis a [pc].
  """

  return (a *pc)**(3/2) *2*np.pi / np.sqrt(G *M_tot *Mo)

def getDistanceFromPeriod(T: float, M_tot: float) -> float:
  """ Calculates the semi-major axis [pc] a of a binary system with
  total mass [M_sun] M_tot with a period T [s].
  """
  
  return (G *M_tot *Mo/(4 *np.pi**2) *T**2)**(1/3) /pc

def getChirpMass(m1: float, m2: float) -> float:
  """ Calculates the chirp mass in the given units."""
  return (m1 *m2)**(3/5) / (m1 +m2)**(1/5)

def getVacuumMergerDistance(m1: float, m2: float, t: float) -> float:
  """ Calculates the distance [pc] at which a binary in vacuum would
  merge after t [s].

  * m1, m2 are the masses [M_sun] of the components.
  * t is the time [s] until merger event.
  """

  return (256 * G**3 *(m1 +m2) *m1 *m2 *Mo**3 / (5 *c**5) *t)**(1/4) /pc

def getVacuumMergerTime(m1: float, m2: float, r: float) -> float:
  """ Calculates the time [s] at which a binary in vacuum would
  merge given that it is at seperation r [pc].

  * m1, m2 are the masses [M_sun] of the components.
  * r is the current seperation [pc] of the binary.
  """

  return (5 *c**5 *(r *pc)**4) / (256 * G**3 *(m1 +m2) *m1 *m2 *Mo**3) # [s]

def getVacuumPhase(fGW: float, m1: float, m2: float) -> float:
  """ Calculates the phase [rad] of the gravitational wave in the vacuum case in
  the Newtonian approximation.

  * m1, m2 are the masses [M_sun] of the two components.
  * fGW is the frequency of the gravitational wave.
  """

  return 1/16 *(c**3 / (np.pi *G *getChirpMass(m1, m2) *Mo *fGW))**(5/3)