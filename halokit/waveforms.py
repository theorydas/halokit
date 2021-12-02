from typing import Union
from .units import *
from .basic import *

import numpy as np

from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import simps, trapz, cumtrapz
from tqdm.notebook import tqdm

def getIntrinsicSNRFromDephase(fGW: np.array, dPhase: np.array, Mc: float, Sn = "LISA") -> float:
  """ Calculates the intrinsic SNR defined by ignoring spatial parameters like distance and inclination of detector
  for a binary system whose environmental effects imprint a dephasing on the gravitational wave signature.

  * fGW is the frequency [Hz] of the gravitational wave signal. This array should end in the f_ISCO of the system.
  * dPhase is the dephasing left until coalescence [rad] of the wave from the vacuum case: dPhase = PhaseV -PhaseSystem
  * Mc is the chirp mass [M_sun] of the binary.
  * Sn describes the noise curve of the detector for which to calculate the intrinsic SNR. Defaults to "LISA", else pass
  an np.array for fGW.
  """
  if Sn == "LISA": Sn = getNoise_LISA(fGW)
  
  # Normalization terms for the SI units of choice.
  frontTerm = 32 *np.pi**(7/3) /c**8 *(G *Mc *Mo)**(10/3)
  psi_V = 1/16 *(c**3 /(np.pi *G *Mc* Mo) )**(5/3)

  # Calculate the phase left until coalescence for the vacuum case. 
  # Assumes that fc = fISCO and fGW properly includes this as its last point.
  Ph_V_to_c = lambda f, fc: psi_V *f**(-5/3) -psi_V *fc**(-5/3)
  Phase_to_c = Ph_V_to_c(fGW, fGW[-1]) -dPhase

  # Calculate the 2nd time derivative for the phase using the inverse from first frequency derivative.
  d2Phasedt2 = 4 *np.pi**2 *fGW *np.nan_to_num(-np.gradient(fGW, Phase_to_c, edge_order = 2))
  
  # Define the part inside of the integral and compute the integration.
  integral = fGW**(4/3) / d2Phasedt2 / Sn
  SNR = np.sqrt(frontTerm *trapz(integral, fGW))
  
  return SNR

def getIntrinsicStrainAmplitude(f: np.array, Phase_to_c: np.array, m1: float, m2: float) -> np.array:
  """ Returns the intrinsic frequency domain amplitude [SI] of the quadrupole strain, unweighted
  by distance or orbital inclination.

  * f is the frequency [Hz] of the gravitational wave.
  * Phase_to_c is the phase of the orbit in fGW.
  * m1, m2 are the masses [M_sun] of the two components.
  """

  Mc = getChirpMass(m1, m2) *Mo # [M_sun]

  # Calculate the second time derivative of the phase as a function of frequency.
  # The minus sign changes phase to coalescence to be proper phase from 0.
  # dPhase_df = -UnivariateSpline(f, Phase_to_c, s = 0).derivative(1)(f)
  dPhase_df = - np.gradient(Phase_to_c, f)
  d2Phase_dt2 = 4 *np.pi**2 *f /dPhase_df
  
  A = 2 *np.pi **(2/3) *(G *Mc)**(5/3) / (c**4) *f**(2/3) *np.sqrt(2 *np.pi / d2Phase_dt2)

  return A

def getIntrinsicStrainPhase(f: np.array, Phase_to_c: np.array) -> np.array:

  # Calculate the first phase derivative of the phase and integrate to find time.
  # The minus sign ensures that phase_to_c goes to proper phase.
  dPhase_df = -np.gradient(Phase_to_c, f)

  t_f = cumtrapz(dPhase_df/f, f, initial = 0) / (2 *np.pi) # [s]
  t_to_c = t_f[-1] -t_f # Remove last integral to inverse t_f to t_to_c

  Psi = -2 *np.pi *f * t_to_c +Phase_to_c
  # Psi = 2 *np.pi *f *(tc +D/c -t_to_c) +Phase_to_c -phasec -np.pi/4

  # Rotate Psi to be proper
  return Psi -Psi[0] -1

def getStrainPolarisations(f: np.array, Phase_to_c: np.array, m1: float, m2: float, raw: bool = False,
                           D: float = 0, i: float = 0,
                           tc: float = 0, phasec: float = 0) -> np.array:

  A0 = getIntrinsicStrainAmplitude(f, Phase_to_c, m1, m2).astype(float)
  Psi = getIntrinsicStrainPhase(f, Phase_to_c).astype(float)

  # hp = A0/2 *(1 + np.cos(i)**2 ) *np.cos(Psi) #/D
  # hc = A0 *np.cos(i) *np.sin(Psi) #/D
  if raw:
    hp = A0 *np.cos(Psi)
    hc = A0 *np.sin(Psi)

    return hp, hc
  else:
    return (A0, Psi)

def getNoise_LISA(f: float, considerGalacticBackground: bool = True, useFit = False) -> float:
  """ Constructs the one-sided intrument noise curve for LISA Sn(f) [1/Hz] for a
  given frequency [Hz] as defined by Eq. 13 of arXiv:1803.01944 without considering
  added noise by unresolved galactic binaries Sc (see Eq. 14).

  * If considerGalacticBackground (True by default) is set to True, it will also
  include predictions for the unresolved galactic binary contributions.
  * If useFit (False by default) it will use a simpler analytical fit as defined by
  Eq. 1.
  """
  L = 2.5e9 #[m]
  f_star = 19.09e-3 #[Hz]

  P_OMS = (1.5e-11)**2 *(1 + (2e-3/f)**4 )
  P_acc = (3e-15)**2 *(1 + (0.4e-3/f)**2 ) *(1 + (f/8e-3)**4)

  if considerGalacticBackground:
    # Set amplitude and parameters for a 4 year observation period.
    A = 9e-45
    a = 0.138; b = -221; k = 521; g = 1680; fk = 0.00113

    # Seems to overflow the exponential sometimes.
    Sc = A *f**(-7/3) *np.nan_to_num(np.exp(-f**a +b*f*np.sin(k *f))) *(1 +np.tanh(g*(fk -f)))
  else:
    Sc = 0
  
  if useFit:
    Sn = 10/(3 *L**2) *(P_OMS +4 *P_acc / (2 *np.pi *f)**4 ) *(1 +6/10 *(f/f_star)**2)
  else:
    Sn = 10/(3 *L**2) *(P_OMS +2 *(1 +np.cos(f/f_star)**2) *P_acc / (2 *np.pi *f)**4 ) *(1 +6/10 *(f/f_star)**2)

  return Sn +Sc

def getLongMismatch(f: np.array, h1: list, h2: list, Sn: np.array, skip: int = 1, verbose: bool = False) -> float:
  """
  Calculates the mismatch between two strains h1, h2 in the frequency domain given a one-sided
  detector noise sensitivity Sn.

  * h1, h2 should be given in the form (A, Psi) each.
  """
  # Decompose the input.
  A1, Psi1 = h1
  A2, Psi2 = h2

  # Calculate the normalizations
  norm1 = np.sqrt(4 *simps(A1**2 / Sn, f))
  norm2 = np.sqrt(4 *simps(A2**2 / Sn, f))

  overlap = 0
  iterator = range(0, len(f) -1, skip)
  if verbose:
    iterator = tqdm(iterator)
  
  # Define a fast 2-point linear interpolator.
  interp2P = lambda y0, y1, x0, x1, x: y0 + (x -x0) *(y1 -y0)/(x1 -x0)

  for i in iterator:
    # Build a tighter grid.
    if i +skip > len(f) -1: break

    f_tight = np.linspace(f[i], f[i +skip], 1000)
    
    # Interpolate quantities to that grid.
    A1_ = interp2P(A1[i], A1[i +1], f[i], f[i +1], f_tight)
    Psi1_ = interp2P(Psi1[i], Psi1[i +1], f[i], f[i +1], f_tight)
    A2_ = interp2P(A2[i], A2[i +1], f[i], f[i +1], f_tight)
    Psi2_ = interp2P(Psi2[i], Psi2[i +1], f[i], f[i +1], f_tight)
    Sn_ = interp2P(Sn[i], Sn[i +1], f[i], f[i +1], f_tight)

    # Can we also get a metric that checks if cos(PSi) is not sampled tightly enough and alerts us?

    # Calculate overlap.real/4 for that part.
    overlap += trapz(A1_ *A2_ *np.cos(Psi2_- Psi1_) /Sn_ , f_tight)

  # Normalize the overlap for the two functions.
  overlap = 4 *overlap/norm1/norm2

  return max(0, min(1, 1 -overlap)) # Constraint between 0 and 1 for the shake of numerical computations

def buildPhase(f: np.array, dPhase: np.array, m1: float, m2: float):
  """ Calculates the phase left until coalescence given a dephase from the
  vacuum case.

  * m1, m2 are the masses [M_sun] of the binary
  * f is the frequency [Hz] of the gravitational wave
  * dPhase is the dephasing [rad] in the frequency domain f.
  """
  # Calculate the phase until coalescence for the vacuum case.
  PhaseV_to_c = getVacuumPhase(np.atleast_2d(f), np.atleast_2d(m1), np.atleast_2d(m2)).astype(float)
  PhaseV_to_c = PhaseV_to_c -PhaseV_to_c[:, -1] # Remove last part to get 0 at f_isco

  Phase_to_c = PhaseV_to_c -dPhase

  if np.ndim(f) == 1:
    return Phase_to_c[0]
  else:
    return Phase_to_c

# Old functions
def getTemplatePSD(source: str) -> np.array:
  """ Imports PSDs from sources and stuff"""
  PSD_Source = {"_LIGO": "https://dcc.ligo.org/public/0149/T1800044/005/aLIGODesign.txt", "LIGO": "https://dcc.ligo.org/public/0156/G1801950/001/2017-06-10_DCH_C02_H1_O2_Sensitivity_strain_asd.txt"}
  
  f, psd = np.loadtxt(PSD_Source[source]).T

  return f, psd

def calculatePSD(f, source):
  if source == "White":
    return np.ones(len(f))
  
  fpsd, psd = getTemplatePSD(source)
  sigma = psd.std() # I get sigma to help with the fit below.

  # Sets up a quick interpolating scheme to get data at f.
  fit = UnivariateSpline(fpsd, psd/sigma, k = 1, s = 0)
  # interpolate this to fpsd

  psd = fit(f)*sigma

  return psd

def dotProduct(h1: np.ndarray, h2: np.ndarray, t: np.ndarray, PSD = None) -> np.ndarray:
  """ Calculates the dot product between two complex waveforms over the
  range t in the time domain. If PSD is specified the dot product will be
  calculated in the frequency domain instead.
  
  The product integration uses the simpsons method."""

  if np.ndim(t) == 1:
    if type(PSD) == np.ndarray: # If PSD exists as an array, calculate the freq domain product.
      integral = simps(h1.conj() *h2/PSD, t)

      return 4 *integral.real
    else:
      return simps(h1.conj() *h2, t)
  else:
    Integrals = []

    if type(PSD) == np.ndarray: # If PSD exists as an array, calculate the freq domain product.
      for h1_, h2_, t_ in zip(h1, h2, t):
        Integrals.append(simps(h1_.conj() *h2_/PSD, t_))
      
      Integrals = np.array(Integrals)

      return 4 *Integrals.real
    else:
      for h1_, h2_, t_ in zip(h1, h2, t):
        Integrals.append(h1_.con() *h2_, t_)
      
      return np.array(Integrals)
  
def normalizeWaveform(h: np.ndarray, t: np.ndarray, PSD = None) -> np.ndarray:
  """ Normalizes a waveform such that its own dot product is equal to one.
  By defailt, assumes time domain product but if PSD is given calculates in
  the frequency domain."""

  norm = np.sqrt(dotProduct(h, h, t, PSD))
  # Reshape to match vertical vector notation.
  norm = norm.reshape(-1, 1)
  
  return h/norm

def getShortMismatch(h1, h2, f, psd):
  # Choose a noise PSD
  PSD = calculatePSD(f, psd)

  # Normalise the waveforms for the given noise.
  h1 = normalizeWaveform(h1, f, PSD)
  h2 = normalizeWaveform(h2, f, PSD)

  # Calculate the mismatch as 1 -normalised dot product.
  # Return a maximum of 1 (may actually be more due to numerical errors)
  return min(1, 1 -dotProduct(h1, h2, f, PSD).real)

def getStrainFromDephase_(f: np.array, dPhase: np.array, m1: float, m2: float, shouldInterpoalte: bool = True):
  # Interpoalte f to higher resolution.

  if shouldInterpoalte:
    # Interpolate dPhase and make tighter grid.
    f_ = np.logspace(np.log10(np.min(f)), np.log10(np.max(f)), 1000000)

    dPhase_ = interp1d(f, dPhase, bounds_error = False, fill_value = "extrapolate")(f_)

  Phase_to_c = getVacuumPhase(f_, m1, m2) -dPhase_
  Phase_f = Phase_to_c[0] -Phase_to_c # Proper Phase from 0

  hp, hc = getStrainPolarisations(f_, Phase_f, m1 = m1, m2 = m2, D = 0, i = 0)
  
  return hp +1j *hc

getStrainFromDephase = lambda F, DPHASE, M1, M2: np.array([getStrainFromDephase_(f_, Dph_, m1_, m2_) for f_, Dph_, m1_, m2_ in tqdm(zip(F, DPHASE, M1, M2)) ])
getShortMismatch = lambda h_true, h_pred, freq: np.array([getShortMismatch(h_true_, h_pred_, freq_, "White") for h_true_, h_pred_, freq_ in tqdm(zip(h_true, h_pred, freq))]).T[0].astype(complex).real