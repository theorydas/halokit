from .units import *

import numpy as np
from scipy.special import gamma, hyp2f1

def getRsp(gamma_sp: float, m1: float, rho_sp: float) -> float:
  """ Calculates the characteristic radius [pc] of the spike
  assuming a power law distribution.
   
  * gamma_sp is the slope of the power law.
  * m1 is the mass [M_sun] of the larger component in the binary.
  * rho_sp [Mo/pc3] is the normalisation density of the spike.

  The calculations are based on Equation 2.2 of arxiv.org/abs/2002.12811.
  """
  return ((3 -gamma_sp) *(m1 *Mo)/(2 *np.pi *(rho_sp *Mo/pc**3) *5**(3 -gamma_sp)))**(1/3) /pc

def rho_spike(r: float, gamma_sp: float, rho_sp: float, m1: float) -> float:
  """ Assumes a powerlaw dark matter spike, and calculates its density [kg/m3]
  at a given distance r [pc] from the center.

  * r is the distance [pc] from the center at which the distribution should be calculated.
  * gamma_sp is the slope of the power law.
  * m1 is the mass [M_sun] of the larger component in the binary.
  * rho_sp [M_sun/pc3] is the normalisation density of the spike.

  The calculations are based on half of Equation 2.1 of arxiv.org/abs/2002.12811.
  """

  # r_sp = ((3 -gamma_sp) *m1/(2 *np.pi *rho_sp *5**(3 -gamma_sp)))**(1/3)
  r_sp = getRsp(gamma_sp, m1, rho_sp) # [pc]
  
  return (rho_sp *Mo/pc**3) *(r_sp/r)**gamma_sp # [kg/m3]

def rho_spike6(r: float, gamma_sp: float, rho6: float) -> float:
  """ Assumes a powerlaw dark matter spike, and calculates its density [kg/m3]
  at a given distance r [pc] from the center.

  * r is the distance [pc] from the center at which the distribution should be calculated.
  * gamma_sp is the slope of the power law.
  * rho6 [M_sun/pc3] is the density of the spike at distance r6 = 1e-6 pc.

  The calculations are based on half of Equation 3 of arxiv.org/abs/2108.04154.
  """
  r6 = 1e-6

  return (rho6 *Mo/pc**3) *(r6/r)**gamma_sp

def getRho6FromSpike(rho_spike: float, gamma_sp: float, m1: float) -> float:
  """ A conversion from the spike density normalisation rho_spike [M_sun/pc3] to rho6 [M_sun/pc3].

  * rho_spike is the density normalisation of the spike.
  * gamma_sp is the slope of the density distribution.
  * m1 is the mass [M_sun] of the central(largest) black hole in the binary system.
  """
  r6 = 1e-6 # [pc]
  rsp = getRsp(gamma_sp, m1, rho_spike)

  return rho_spike *(r6/rsp)**gamma_sp

def getRhoSpikeFrom6(rho6: float, gamma_sp: float, m1: float) -> float:
  """ A conversion from the spike density normalisation rho6 [M_sun/pc3] to rho_spike [M_sun/pc3].

  * rho6 is the density of the spike at distance r6 = 1e-6 pc.
  * gamma_sp is the slope of the density distribution.
  * m1 is the mass [M_sun] of the central(largest) black hole in the binary system.
  """
  r6 = 1e-6 *pc
  A = (3 - gamma_sp) *0.2**(3-gamma_sp) *(m1 *Mo)/(2 *np.pi)
  A = A **(gamma_sp/3)

  return ((rho6 *Mo/pc**3)/A *r6 **gamma_sp)**(1/(1 -gamma_sp/3)) /(Mo/pc**3)

def getKsiCDM(q: float, gamma_sp: float) -> float:
  """ Returns the fraction of particles ksi that move slower than the orbital velocity of the secondary black hole.

  * q is the mass ratio of the two components m2/m1.
  * gamma_sp is the slope of the dark matter distribution.
  """
  f = 1/2 *(1 +q)

  ksi = 4/3/np.sqrt(np.pi) *gamma(gamma_sp +1)/gamma(gamma_sp -1/2)\
  *hyp2f1(3/2, 3/2 -gamma_sp, 5/2, f) *f**(3/2)
  
  return ksi

def getStaticBreakFrequency(m1: float, m2: float, gamma_sp: float, rho_sp: float) -> float:
  """ Returns the break (equality) frequency [Hz] as defined by the matching of the gravitational
  and dynamic friction energy losses of a static power law dark matter spike in Equation 15
  of arxiv.org/pdf/2108.04154.pdf.

  * m1, m2 are the masses [M_sun] of the two components.
  * gamma_sp is the slope of the dark matter distribution.
  * rho_sp is the density [M_sun/pc3] normalisation of the spike.
  """
  Lambda = np.sqrt(m1/m2)
  ksi = getKsiCDM(m2/m1, gamma_sp)

  # The size of the spike
  rsp = getRsp(gamma_sp, m1, rho_sp) *pc # [m]

  c_f = (5 *c**5)/(8 *(m1 *Mo)**2) *np.pi**(2/3 *(gamma_sp -4)) *G**(-2/3 -gamma_sp/3)\
  * (m1 *Mo +m2 *Mo)**(1/3 -gamma_sp/3) *rsp**gamma_sp *ksi *(rho_sp *Mo/(pc)**3) *np.log(Lambda)
  
  f_eq = c_f **(3/(11 -2 *gamma_sp)) # [Hz]

  return f_eq

def getDynamicBreakFrequency(m1: float, m2: float, gamma_sp: float) -> float:
  """ Returns the break frequency [Hz] as fitted in Equation 35 of
  arxiv.org/pdf/2108.04154.pdf for the phenomenological description of a dynamically
  evolving CDM dark matter spike.

  * m1, m2 are the masses [M_sun] of the two components.
  * gamma_sp is the slope of the dark matter distribution.
  """
  # The empirical parametres fit from ~80 simulations.
  a1 = 1.4412; a2 = 0.4511; b = 0.8163; z = -0.4971; gamma_r = 1.4396

  fb = b *(m1/1000)**(-a1) *(m2)**(a2) *(1 +z *np.log(gamma_sp/gamma_r)) # [Hz]

  return fb