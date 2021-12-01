from .dynamical_friction._evolution import dr2dt
from . import units

from scipy.integrate import cumtrapz
from tqdm.notebook import tqdm
import numpy as np

def lolol():
  return units.G

def getDephasingUntilMerger_Backbone(m1: float, m2: float, gamma_sp: float, rho_sp: float,
                                fGW: float, isStatic: bool = True) -> float:
  """ An analytical (and empirical) model introduced in arxiv.org/abs/2108.04154 for the static
  (or dynamically evolving) dark matter spike. This function returns the orbital dephasing [rad]
  until coalescence as a function of the gravitational wave frequency [Hz] between a vacuum system
  and a system with dynamical friction in either assumption.

  This function creates a power law spike using the HaloFeedback module.
  This function is wrapped by getDephasingUntilMerger().

  * m1, m2 are the masses [M_sun] of the two components.
  * gamma_sp is the slope of the dark matter distribution.
  * rho_sp is the density [M_sun/pc3] normalisation of the spike.
  * fGW is the frequency [Hz] of the gravitational waves.
  * isStatic is a boolean that controls wether to calculate the dephasing for a static or dynamically evolving spike. 
  """
  f_eq = getStaticBreakFrequency(m1, m2, gamma_sp, rho_sp) # [Hz]

  if not isStatic:
    ft = getDynamicBreakFrequency(m1, m2, gamma_sp) # [Hz]
    
    gamma_e = 5/2 # The final density slope of the shell model.

    heta = (5 +2 *gamma_e) / (16 - 2 *gamma_sp) *(f_eq/ft) **(11/3 -2*gamma_sp/3)
    lambd = (11 -2 *(gamma_sp +gamma_e))/3
    theta = 5/(2 *gamma_e)
  else:
    heta = 1
    lambd = 0
    theta = 5/(11 -2 *gamma_sp)

    ft = f_eq

  y = fGW/ft

  dark_weight = np.array([hyp2f1(1, theta, 1 +theta, -_y**(-5/(3 *theta))) for _y in y])
  dark_weight = 1 -heta *y**(-lambd) *(1 -dark_weight)

  Phase_Vacuum = getVacuumPhase(fGW, m1, m2)
  Phase = Phase_Vacuum *dark_weight

  Dph = Phase_Vacuum -Phase

  return Dph

def getDephasingUntilMerger(m1: float, m2: float, gamma_sp: float, rho_sp: float, fGW: float, isStatic: bool = True) -> float:
  """ An analytical (and empirical) model introduced in arxiv.org/abs/2108.04154 for the static
  (or dynamically evolving) dark matter spike. This function returns the orbital dephasing [rad] as
  a function of the gravitational wave frequency [Hz] between a vacuum system and a system with
  dynamical friction in either assumption.

  This function creates a power law spike using the HaloFeedback module.
  This function serves as a wrapped for getDephasingUntilMerger_Backbone() which actually computes the dephasing.

  * m1, m2 are the masses [M_sun] of the two components.
  * gamma_sp is the slope of the dark matter distribution.
  * rho_sp is the density [M_sun/pc3] normalisation of the spike.
  * fGW is the frequency [Hz] of the gravitational waves.
  * isStatic is a boolean that controls wether to calculate the dephasing for a static or dynamically evolving spike.
  """

  if type(m1) == np.ndarray:
    Dph = getDephasingUntilMerger_Backbone(m1[0], m2[0], gamma_sp[0], rho_sp[0], np.atleast_2d(fGW[0]), isStatic)

    if len(m1) == 1:
      return Dph

    dephases = []
    dephases.append(Dph.reshape(1, -1))

    for _m1, _m2, _gamsp, _rhosp, _fGW in tqdm(zip(m1[1:], m2[1:], gamma_sp[1:], rho_sp[1:], fGW[1:])):
      _Dph = getDephasingUntilMerger_Backbone(_m1, _m2, _gamsp, _rhosp, np.atleast_2d(_fGW), isStatic)

      dephases.append(_Dph.reshape(1, -1))

      # Dph = np.hstack([Dph, _Dph]) # This vstack may be slow, should move outisde loop?
    
    return np.vstack(dephases)
  else:
    Dph = getDephasingUntilMerger_Backbone(m1, m2, gamma_sp, rho_sp, fGW, isStatic)
    
  return Dph

def sampleDynamicDephasing(datasetSize: int, f: np.array, m1 = [1e3, 1e5], m2 = [1, 10], gamma_sp = [2.25, 2.45], rho_sp = [20, 2000], scaleFrequencyToISCO = True):
  M1 = np.random.random(datasetSize) *(m1[1] -m1[0]) +m1[0]
  M2 = np.random.random(datasetSize) *(m2[1] -m2[0]) +m2[0]
  Gamma_sp = np.random.random(datasetSize) *(gamma_sp[1] -gamma_sp[0]) +gamma_sp[0]
  Rho_sp = np.random.random(datasetSize) *(rho_sp[1] -rho_sp[0]) +rho_sp[0]

  if scaleFrequencyToISCO:
    # Rescale the input frequency in f_ISCO units.
    fisco = 2/getPeriodFromDistance(getRisco(M1 +M2), M1 +M2)
    f = np.logspace(np.log10(np.min(f) *fisco), np.log10(np.max(f) *fisco), len(f)).T

  # Calculate the empirical expressions for the dephasing
  dPhase = getDephasingUntilMerger(m1 = M1, m2 = M2, gamma_sp = Gamma_sp, rho_sp = Rho_sp, fGW = f, isStatic = False)
  dPhase = dPhase - dPhase[:, -1].reshape(-1, 1) # Remove last part. MUST correspond to f_ISCO

  # Construct the input vector that is fed into surrogate models
  input = np.vstack([M1, M2, Gamma_sp, Rho_sp]).T

  if scaleFrequencyToISCO:
    return f, dPhase.astype(float), input
  else:
    return dPhase.astype(float), input

def getDephasingFromDensity(r, rho, m1, m2):
  # Construct the quantity inside of the integral
  fGW = 2/getPeriodFromDistance(r, m1 +m2)
  # integral = fGW *(1/dr2dt(r, m1, m2, rho) -1/dr2dt(r, m1, m2, 0))

  drdtGW, drdtDF = dr2dt(r, m1, m2, rho, separate = True)
  integral = fGW *-drdtDF/drdtGW/(drdtGW +drdtDF)

  dPhase = 2 *np.pi *cumtrapz(integral, r, initial = 0)

  # Calculate the time evolution for the binary
  t = cumtrapz(1/dr2dt(np.flip(r), m1, m2, rho), np.flip(r), initial = 0)

  return t, np.flip(fGW)/2, np.flip(dPhase)