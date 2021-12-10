import HaloFeedback
from .units import *

import numpy as np


def dEdt_GW(r2: float, m1: float, m2: float) -> float:
  """ An analytical approximation for calculating the rate of energy loss [kg*m2/s3]
  for a binary system due to the radiation of gravitational waves.

  * r2 is the seperation [pc] between the two components.
  * m1, m2 are the masses [M_sun] of the two components.
  """
  m1 *= Mo; m2 *= Mo
  r2 = r2.copy() *pc

  return 32 * G**4 *(m1 +m2) *(m1 *m2)**2 /(5 *c**5 *r2**5)

def dEdt_DF(r2: float, m1: float, m2: float, rho_DM_at_r2: float) -> float:
  """ A numerical model for calculating the rate of energy loss [kg*m2/s3]
  for a binary system due to the dynamical friction with any dark matter distribution.

  * r2 is the seperation [pc] between the two components.
  * m1, m2 are the masses [M_sun] of the two components.
  * rho_DM_at_r2 [M_sun/pc3] is the density at distance r2 of dark matter particles
  (which are faster than the orbital velocity at that distance).

  """
  Lambda = np.sqrt(m1/m2)
  u_orb = lambda r: np.sqrt(G*(m1 +m2) *Mo/(r*pc)) # [m/s]

  return 4 *np.pi *(G *m2 *Mo)**2 *rho_DM_at_r2 *Mo/pc**3 *np.log(Lambda)/u_orb(r2)

getRelevantDensity = lambda spike, r: spike.rho(r, np.sqrt(G*(spike.M_BH +spike.M_NS)*Mo/r/pc)/1000)
getRelevantDensity = np.vectorize(getRelevantDensity, excluded = [0])