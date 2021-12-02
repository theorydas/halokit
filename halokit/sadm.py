# Setup imports
from .units import *
from .halos import *

import numpy as np

from scipy.integrate import trapz, simps
from scipy.special import gamma
from scipy.interpolate import interp1d

from tqdm.notebook import tqdm


# Useful functions

def getCoredDensity(mx: float, sigmau: float, t: float) -> float:
  """ Returns the cored density [kg/m3] associated with
  a self annihilating dark matter distribution.

  * mx is the mass [GeV/c2] of the dark matter particle.
  * sigmau is the thermally averaged cross section x relative velocity product [cm3/s].
  * t is an estimate of the time [s] the annihilations are acting.
  """

  return mx *GeVc2 / (sigmau *1e-6 *t)

def getRcored(rho_c: float, rho_sp: float, m1: float, gamma_sp: float) -> float:
  """ Calculates the radius [pc] at which a cored density [kg/m3] would intersect
  an adiabatically grown power-law dark matter spike density.

  * rho_c is the cored density [kg/m3] associated with the dark matter self annihilation.
  * rho_sp is the density [M_sun/pc3] normalization of the dark matter spike.
  * m1 is the mass [M_sun] of the central black hole which the above spike grew around.
  * gamma_sp is the slope of the density.
  """
  Rsp = getRsp(gamma_sp, m1, rho_sp) # [pc]

  return Rsp *(rho_sp/(rho_c /(Mo/pc**3)))**(1/gamma_sp)

def getPotential(m1: float, r: float) -> float:
  """ Calculates the potential of the gravitational field [m2/s2] felt by dark matter
  particles around it.

  * m1 is the mass [M_sun] of the black hole.
  * r is the distance [pc] from that black hole.
  """

  return G *(m1 *Mo) /(r *pc) # [m2/s2]

def getAnnihilationRate(spike, m1: float, weight, s = 0, uf0 = c/5, verbose = False):
  """ Calculates the density contribution of the annihilation rate of a dark matter
  distribution. s-wave is s = 0 and p-wave is s = 2.
  """
  # Set up the density interpolator
  # Todo: Think about not including the bad area of large separations?
  _r_grid = np.logspace(np.log10( G *m1 *Mo/np.max(spike.psi) /pc ), np.log10( G *m1 *Mo/np.min(spike.psi) /pc ), 1000)
  _psi_grid = G *m1 *Mo/(_r_grid *pc)
  rho_ = spike.getRho(spike, _psi_grid)

  if spike.psi_cut != -1:
    rho_[_psi_grid >= spike.psi_cut] = 0

  if s != 0:
    rho_ *= np.nan_to_num(spike.getVelocityDispersion(spike, _psi_grid, rho_))**s
  
  rho_ = interp1d(_r_grid, rho_)

  rate_eps = []
  for eps in tqdm(spike.psi[1:]) if verbose else spike.psi[1:]: # Skip the first to avoid bad results.
    r_grid = np.logspace(np.log10( G *m1 *Mo/np.max(spike.psi) /pc ), np.log10( G *m1 *Mo/eps /pc ), 1000, endpoint = False)
    psi_grid = G *m1 *Mo /(r_grid *pc)
    rho_grid = rho_(r_grid)

    inside = (r_grid *pc)**2 *np.sqrt(psi_grid - eps)
    integral = trapz(rho_grid* inside, r_grid *pc)
    norm = trapz(inside, r_grid *pc) # TODO: Think about calculating analytically when Psi = Gm1/r

    rate = integral/norm
    rate_eps.append(rate)

  # Extrapolate to the first, skipped element:
  # rate0 = rate_eps[1] +(Psi[0] -Psi[1]) *(rate_eps[1] -rate_eps[2])/(Psi[1] -Psi[2])
  rate0 = 1 # It shouldn't matter at ISCO right?, things should be zero there?
  rate_eps = np.insert(rate_eps, 0, rate0)

  return (weight *1e-6/GeVc2) *np.sqrt(2) *rate_eps/ uf0**s

def getSpikeDF(eps, gamma_sp, rho_sp, m1):
  """ Calculates the DF for a powerlaw spike.

  * eps is the potential energy at which to calculate the DF.
  * gamma_sp is the slope ofthe powerlaw.
  * rho_sp is the density [M_sun/pc3] normalization of the spike.
  * m1 is the mass of the central black hole that the spike grew from.
  """
  r_sp = getRsp(gamma_sp, m1, rho_sp) # [pc]

  Gamm = gamma(gamma_sp -1)/gamma(gamma_sp -1/2)
  feps = gamma_sp *(gamma_sp -1) *(rho_sp *Mo/pc**3)\
        *(r_sp *pc/G/m1/Mo)**gamma_sp\
        *Gamm *eps**(gamma_sp -3/2) /(2 *np.pi)**(3/2)
  
  return feps

# The distribution function class

class distributionFunction:
  """
  The distributionFunction class will generate the DF [SI] of the system for a given potential psi [SI] and density rho [SI] as
  well as hold functions for computing interesting quantities like reconstructing densities or velocity dispersions.
  
  * clipNegative defaults to True, and controls wether negative DF values should be replaced by 0 or not.
  * psi_cut is to enforce a cutoff potential after which the DF vanishes.
  
  # TODO: What does feps do? Definitely not required to initialize though.....
  """
  def __init__(self, psi: np.array = None, rho: np.array = None, psi_cut: float = -1, feps: np.array = None, clipNegative = True):
    if psi is None or rho is None:
      raise ValueError("Provide a potential: psi, and density: rho array.")
    
    self.psi = psi
    self.rho_init = rho
    self.psi_cut = psi_cut
    self.clipNegative = clipNegative
    
    # TODO: help
    if feps is None:
      self.feps_init = self.getFeps(self.psi, self.rho_init)
    else:
      if self.psi_cut != -1:
        feps = np.array(feps)
        feps[self.psi >= self.psi_cut/1.001] = 0
      
      feps = interp1d(self.psi, feps, bounds_error = False, fill_value = 0)
      self.feps_init = feps
    
    self.feps = self.feps_init

  def __call__(self, eps):
    """ Returns the distribution function at eps interpolated from the grid it was initialized on.
    """
    return self.feps(eps)

  def updateDistribution(self, df):
    """ Replaces the old interpolator with the updated phase distribution f +df.
    
    # TODO: Probably this depricates feps as input right?
    """
    f_eps = interp1d(self.psi, self.feps(self.psi) +df, bounds_error = False, fill_value = 0)
    self.feps = f_eps

    return None

  def getFeps(self, Psi, rho):
    """ Initializes the distribution function [SI] for a given set of Psi and rho in [Si].
    """
    # Calculate the 2nd derivative assuming the 1st doesn't contribute to the distribution, because of vanishing ISCO density.
    d2rho_dPsi2 = np.gradient(np.gradient(rho, Psi, edge_order = 2), Psi) # [SI]
    d2rho_dPsi2 = interp1d(Psi, d2rho_dPsi2, bounds_error = False, fill_value = 0)

    eps_min = np.min(self.psi)

    f_eps = []
    for eps in tqdm(self.psi): # This used to be Psi.
      # Create a grid for the interpolation range. 1e-10 ~ 0 in linspace.
      # new_grid = np.logspace(np.log10(2 *np.sqrt(eps -eps_min))-1, np.log10(2 *np.sqrt(eps -eps_min)), 1000 *len(self.psi))
      # psi_grid = eps -new_grid**2 /4

      # ==============
      # psi_grid = np.linspace(eps, eps_min, 1000000)
      # new_grid = 2 *np.sqrt(eps -psi_grid)
      # ==============

      new_grid = np.linspace(0, 2 *np.sqrt(eps -eps_min), 100000) # TODO: Is 1m points an overkill? Maybe chanege to 100k to be slightly faster? November 8th 2021
      psi_grid = eps -new_grid**2 / 4

      d2rho_dPsi2_ = d2rho_dPsi2(psi_grid)

      _f = trapz(d2rho_dPsi2_, new_grid) / (np.pi**2 *np.sqrt(8))
      f_eps.append(_f)

    f_eps = np.array(f_eps)
    # Clip negative DF values if required.
    f_eps = f_eps.clip(0) if self.clipNegative else f_eps

    if self.psi_cut != -1:
      """ self.psi_cut must be included in original psi to avoid problems.
      """
      # TODO: This is a bad implementation, but forcefully set some ISCO thereshould.
      f_eps[self.psi >= self.psi_cut/1.001] = 0

    # TODO: Add a boundary error outside.
    f_eps = interp1d(Psi, f_eps, bounds_error = False, fill_value = 0)

    return f_eps
  
  def getRho(self, Psi: float, v_cut = -1, vmin = 0):
    # If not specified, integrate until maximum bound velocity.
    if v_cut == -1: v_cut = np.sqrt(2 *Psi)

    # u_grid = np.linspace(0, v_cut, 5000) # should be 100000
    # eps_grid = Psi - u_grid**2 /2

    # rho = 4 *np.pi *trapz(u_grid**2 *self.feps(eps_grid), u_grid)
    # ==========
    eps_grid = np.linspace(Psi -v_cut**2/2, Psi, 10000)
    
    rho = 4 *np.pi *np.sqrt(2) *trapz(np.sqrt(Psi -eps_grid) *self.feps(eps_grid), eps_grid)
    # ==========

    # if self.psi_cut != -1:
    #   rho = np.array([0]) if Psi >= self.psi_cut else rho
    
    # Remove unphysical negative densities if present.
    return rho.clip(0)

  def getAverageVelocity(self, Psi):
    """ Calculates the average velocity [m/s] at potential Psi.
    """
    u_grid = np.linspace(0, np.sqrt(2 *Psi), 50000)

    integral = trapz(u_grid**3 *self.feps(Psi -u_grid**2 /2), u_grid)
    return 4 *np.pi/self.getRho(self, Psi) *integral
  
  def getAverageVelocitySquared(self, Psi):
    """ Calculates the average velocity squared [m2/s2] at potential Psi.
    """
    u_grid = np.linspace(0, np.sqrt(2 *Psi), 50000)

    integral = trapz(u_grid**4 *self.feps(Psi -u_grid**2 /2), u_grid)
    return 4 *np.pi/self.getRho(self, Psi) *integral
  
  def getVelocityDispersion(self, Psi, Rho = -1):
    """ Calculates the standard deviation / velocity dispersion [m2/s2] at potential Psi.
    """
    
    if Rho == -1:
      Rho = self.getRho(self, Psi)

    u_grid = np.linspace(0, np.sqrt(2 *Psi), 50000)

    Average = 4 *np.pi *trapz(u_grid**3 *self.feps(Psi -u_grid**2 /2), u_grid)/Rho
    Squared = 4 *np.pi *trapz(u_grid**4 *self.feps(Psi -u_grid**2 /2), u_grid)/Rho

    return np.sqrt(Squared - Average**2)

    # return np.sqrt(self.getAverageVelocitySquared(self, Psi) -self.getAverageVelocity(self, Psi)**2)
  
  def getAverageVelocityDifference(self, Psi):
    """ Calculates the average difference of velocities [m/s] between particles at potential Psi.
    #TODO: This is extremely slow, how to optimize?
    """
    # Set up a common base integral for the double integration.
    u_grid = np.linspace(0, np.sqrt(2 *Psi), 1000)
    
    integral = []
    for u1 in u_grid:
      integral_ = simps(u1**2 *u_grid**2 *self.feps(Psi -u1**2 /2) *self.feps(Psi -u_grid**2 /2) *np.abs( u1 -u_grid ), u_grid)
      integral.append(integral_)
    integral = simps(integral, u_grid)

    return 16 *np.pi**2 /self.getRho(self, Psi)**2 *integral

  getRho = np.vectorize(getRho)  
  getAverageVelocity = np.vectorize(getAverageVelocity)
  getAverageVelocitySquared = np.vectorize(getAverageVelocitySquared)
  getVelocityDispersion = np.vectorize(getVelocityDispersion)
  getAverageVelocityDifference = np.vectorize(getAverageVelocityDifference)