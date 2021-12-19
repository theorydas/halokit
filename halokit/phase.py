from typing import Tuple
from .units import *
from .basic import getPeriodFromDistance
from .evolution import dr2dt

import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

def getPhaseFromFrequency(fGW: np.array, t: np.array, returnRaw: bool = False):
  """ Returns the cummulative phase [rad] along time as as an interpolant function of the gravitational
  wave frequency [Hz] of that merger.

  * fGW is the gravitational wave frequency [Hz] of the merger.
  * t is the time [s] corresponding to that frequency.
  * method should be set by default to 0 for an intuitive integration with minimised
  errors by default.
  * returnRaw controls if an interpolant should be returned or simply the phase
  as an array of the same size as the input frequencies.
  """
  
  Phase = 2 *np.pi *cumtrapz(fGW *np.gradient(t, fGW, edge_order = 2), fGW, initial = 0)

  if returnRaw:
    return Phase
  else:
    Phase_fit = interp1d(fGW, Phase, bounds_error = False, fill_value = "extrapolate")
    return Phase_fit

def getDephasingFromFrequencyEvolution(t0: np.array, f0: np.array, t: np.array, f: np.array, fGWc: float) -> np.array:
  """ Builds the dephasing of the system with a frequency evolution f(t) with respect to another f0(t0)"""
  fGW0 = 2 *f0
  fGW = 2 *f

  Phase0 = getPhaseFromFrequency(fGW0, t0)
  Phase = getPhaseFromFrequency(fGW, t)

  Phase0_c = lambda f: Phase0(fGWc) - Phase0(f)
  Phase_c = lambda f: Phase(fGWc) - Phase(f)

  # Calculate the dephasing until merger with respect the vacuum case
  dPhase = Phase0_c(fGW) -Phase_c(fGW)

  return dPhase

def getDephasingFromDensity(r: np.array, rho: np.array, m1: float, m2: float) -> Tuple[np.array, np.array, np.array]:
  """ Creates the time and orbital frequency evolution of the dephasing until coalescence induced
  on the binary from its vacuum evolution, because of an effective density profile rho(r).
  
  * r is the binary separation [pc] for which the effective density is rho(r).
  * rho is the effective density [M_sun/pc3] at a given binary separation.
  * m1, m2 are the masses [M_sun] of the binary components.
  
  Returns t, fGW/2, dPhase
  """
  # Construct the quantity inside of the integral
  fGW = 2/getPeriodFromDistance(r, m1 +m2)

  drdtGW, drdtDF = dr2dt(r, m1, m2, rho, separate = True)
  integrand = fGW *-drdtDF/drdtGW/(drdtGW +drdtDF)

  dPhase = 2 *np.pi *cumtrapz(integrand, r, initial = 0)

  # Calculate the time evolution for the binary
  t = cumtrapz(1/dr2dt(np.flip(r), m1, m2, rho), np.flip(r), initial = 0)

  return t, np.flip(fGW)/2, np.flip(dPhase), # [s, Hz, rads]