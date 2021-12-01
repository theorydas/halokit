from .units import *

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