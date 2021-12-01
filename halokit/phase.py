from HaloToolkit.units import *
from HaloToolkit.basic import *
from HaloToolkit.dynamical_friction._evolution import dr2dt

import numpy as np
from scipy.integrate import cumtrapz, trapz
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

def getDephasingFromFrequencyEvolution(t_V: np.array, f_V: np.array, t: np.array, f: np.array, fGWc: float) -> np.array:
  fGW_V = 2 *f_V
  fGW = 2 *f

  Phase_V = getPhaseFromFrequency(fGW_V, t_V)
  Phase = getPhaseFromFrequency(fGW, t)

  Phase_Vc = lambda f: Phase_V(fGWc) - Phase_V(f)
  Phase_c = lambda f: Phase(fGWc) - Phase(f)

  # Calculate the dephasing until merger with respect the vacuum case
  dPhase = Phase_Vc(fGW) -Phase_c(fGW)

  return dPhase