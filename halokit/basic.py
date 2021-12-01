from .units import *

import numpy as np

def E_orb(r2: float, m1: float, m2: float) -> float:
  """ Calculates the orbital energy [kg*m2/s2] of a binary system.

  * r2 is the seperation [pc] of the two components.
  * m1, m2 are the masses [M_sun] of the components.
  """

  return - G *m1 *m2 *Mo**2 /(2 *r2 *pc)

def getRisco(m: float) -> float:
  """ Calculates the radius [pc] of the Innermost Stable Circular Orbit
  for a massive object of mass m [M_sun].
  """
  
  return 6 *G *m *Mo/c**2 /pc # Turn m into pc

def getFisco(m1: float, m2: float) -> float:
  """ Calculates the orbital frequency [Hz] of the Innermost Stable Circular Orbit
  for a massive object of mass m1 [M_sun] in a binary system with another object m2 [M_sun].
  """

  return 1/getPeriodFromDistance(getRisco(m1), m1 +m2)

def getPeriodFromDistance(r2: float, M_tot: float) -> float:
  """ Calculates the Kepplerian period [s] of binary system with
  total mass [M_sun] M_tot at a seperation r2 [pc].
  """

  return (r2 *pc)**(3/2) *2*np.pi / np.sqrt(G *M_tot *Mo)

def getDistanceFromPeriod(T: float, M_tot: float) -> float:
  """ Calculates the seperation [pc] r2 of a binary system with
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