# from halokit.units import *
from .units import *
from .basic import *
from . import HaloFeedback

from scipy.interpolate import interp1d
from tqdm.notebook import tqdm
import numpy as np

def dr2dt(r2: float, m1: float, m2: float, rho_DM_at_r2: float, separate: bool = False) -> float:
  """ Calculates the time derivative [pc/s] of the seperation between the
  two components in a binary (m1, m2) due to gravitational and dynamical
  friction (with a dark matter distribution) energy losses as defined by Equation 2.6
  of arxiv.org/abs/2002.12811.

  * r2 is the seperation [pc] between the two components.
  * m1, m2 are the masses [M_sun] of the two components.
  * rho_DM_at_r2 [M_sun/pc3] is the effective density at distance r2 of dark
  matter particles (which are faster than the orbital velocity at that distance).
  
  If separate, returns a list of two elements for the gravitational and dynamic friction part.
  """

  M = m1 +m2
  Lambda = np.sqrt(m1/m2)

  c_ = c /pc # [pc/s]
  G_ = G *Mo/pc**3 # [pc3/M_sun/s2]

  dr2dt_GW = - 64 * G_**3 *M *m1 *m2/(5 *c_**5 *r2**3)
  dr2dt_DF = - 8 *np.pi *np.sqrt(G_/M) *(m2/m1) *np.log(Lambda) *rho_DM_at_r2 *r2**(5/2)

  if separate:
    return dr2dt_GW, dr2dt_DF # [pc/s]
  else:
    return dr2dt_GW +dr2dt_DF # [pc/s]

def evolveBinary(m1: float, m2: float, Spike: HaloFeedback.PowerLawSpike, r2_0: float,
                     dtOverT: int = 1000, maxIterations: int = 100000, r_rhoDM = None,
                     isStatic: bool = False, verbose: bool = True, uncertainty: float = 0, r_drop: float = 6e-9, r_law = 2.5, tmin = 0):
  """ Evolves the seperation between the two compact objects of the merger in the prescence
  of a dark matter distribution assuming energy losses due to gravitational radiation and
  dynamical friction with the dark matter distribution. The distribution itself is updated
  using HaloFeedback, due to interactions with the components and therefore depleted. This
  function will then return at each timestep [s], the seperation r2 [pc], the frequency [Hz]
  of the merger and the energy radiated due to dynamical friction [kg*m2/s2].

  * m1, m2 are the masses [M_sun] of the two components.
  * Spike is a HaloFeedback dark matter distribution.
  * r2_0 is the initial seperation [pc].
  * dtOverT is the largest timestep [Periods] possible whenever possible.
  * maxIterations sets the maximum number of steps for the code.
  * isStatic is a boolean that controls wether the dark matter distribution
  should be updated or not.
  * verbose when set to False will hide seperation and timestep updates as well
  as the tqdm functionality for keeping track of time.
  
  Returns t, r, forbit, rho_eff
  """
  r2 = np.array([r2_0]) # [pc]
  Risco = getRisco(m1) # [pc]

  # dE_DF = []
  dt = []
  rho_at_r2 = []
  dtOverT_ = 1
  
  iters = range(maxIterations)
  if verbose:
    iters = tqdm(iters)

  if isStatic:
    # In the case of a static spike, calculate the densities at a tight grid first
    # and just pick them up from the interpolator when needed.
    if r_rhoDM is not None:
      _r, _rho = r_rhoDM
    else:
      print("Seting up the density sampler.")
      _r = np.logspace(np.log10(Risco), np.log10(r2_0), 10000)

      _rho = np.array([Spike.rho(_r0, np.sqrt(G*(m1 +m2) *Mo/(_r0 *pc))/1000) for _r0 in _r]) # [M_sun/pc3]
      print("Density sampler completed. Evolution begins.")
    
    rho_ = interp1d(_r, _rho, bounds_error = False, fill_value = "extrapolate")
    
  for i in iters:
    if verbose and i % 1000 == 1: print(r2[-1], dtOverT_)

    # The timestep is a fraction of the orbit. Should drastically decrease at
    # small seperations to retain high resolution at larger frequencies.
    T = getPeriodFromDistance(r2[-1], m1 +m2)
    dtOverT_ = dtOverT *(min(1, r2[-1]/r_drop))**r_law
    # dtOverT_ = max(0.05, dtOverT *(min(1, T/150))**2)
    dtOverT_ = max(dtOverT_, tmin)
    dt.append(T *dtOverT_)
    
    v_orb = np.sqrt(G*(m1 +m2) *Mo/(r2[-1] *pc))/1000 # [km/s]
    
    if not isStatic:
      # Update the spike distribution for O(dt)
      df1 = Spike.delta_f(r2[-1], v_orb = v_orb, dt = dt[-1], v_cut = v_orb)
      Spike.f_eps += df1
      
      # df2 = Spike.delta_f(r2[-1], v_orb = v_orb, dt = dt[-1], v_cut = v_orb)
      # Spike.f_eps += 0.5 * (df2 - df1)

    # Calculate the new effective density distribution based on updated phase distribution.
    if not isStatic:
      rho_DM = Spike.rho(r2[-1], v_orb) # [M_sun/pc3]
    else:
      rho_DM = rho_(r2[-1]) # [M_sun/pc3]
    
    h = dt[-1]
    if isStatic:

      # Update seperation of binary from the equations of motion
      k1 = dr2dt(r2[-1], m1, m2, rho_DM_at_r2 = rho_(r2[-1]))
      # k2 = dr2dt(r2[-1] +h *k1/2, m1, m2, rho_DM_at_r2 = rho_(r2[-1] +h *k1/2))
      # k3 = dr2dt(r2[-1] +h *k2/2, m1, m2, rho_DM_at_r2 = rho_(r2[-1] +h *k2/2))
      # k4 = dr2dt(r2[-1] +h *k3, m1, m2, rho_DM_at_r2 = rho_(r2[-1] +h *k3))

      # r2 = np.append(r2, r2[-1] +h/6 *( k1 +2*k2 +2 *k3 +k4))
      r2 = np.append(r2, r2[-1] +h *k1)
    else:
      k1 = dr2dt(r2[-1], m1, m2, rho_DM_at_r2 = rho_DM)
      r2 = np.append(r2, r2[-1] +h *k1)

    rho_at_r2.append(rho_DM)
    
    # Break the propagation if we reach the Innermost Stable Circular Orbit (ISCO)
    if r2[-1] < Risco: break
  
  r2 = r2[:-1] # Drop the last value which was updated before the code terminated.
  T = getPeriodFromDistance(r2, m1 +m2)
  # dE_DF = np.array(dE_DF) *1e6 *Mo /(dt/T) # Unit conversion from HaloFeedback output + #orbits normalisation.
  t = np.cumsum(dt)
  rho_at_r2 = np.array(rho_at_r2) # [M_sun/pc3]

  return t, r2, 1/T, rho_at_r2#, vs, v2s

def evolveBinaryVacuum(m1: float, m2: float, r2_0: float, dtOverT: int = 1000, maxIterations: int = 100000, r_drop = 6e-9, r_law = 3, tmin = 0):
  """ Evolves the seperation between the two compact objects of the merger in vacuum assuming
  energy losses due to gravitational radiation This function will then return at each timestep [s],
  the seperation r2 [pc] and the frequency [Hz] of the merger.

  * m1, m2 are the masses [M_sun] of the two components.
  * r2_0 is the initial seperation [pc].
  * dtOverT is the largest timestep [Periods] possible whenever possible.
  * maxIterations sets the maximum number of steps for the code.
  
  Returns t, r, forbit.
  """
  r2 = np.array([r2_0]) # [pc]
  Risco = getRisco(m1) # [pc]

  dt = []
  dtOverT_ = 1
  
  for i in tqdm(range(maxIterations)):    
    if i % 1000 == 1: print(r2[-1], dtOverT_)

    # Calculate time step
    T = getPeriodFromDistance(r2[-1], m1 +m2)
    # dtOverT_ = dtOverT *(min(1, T/150))**2
    dtOverT_ = dtOverT *(min(1, r2[-1]/r_drop))**r_law
    dtOverT_ = max(dtOverT_, tmin)

    dt.append(T *dtOverT_)

    # Update seperation of binary through equations of motion
    dr2 = dr2dt(r2[-1], m1, m2, rho_DM_at_r2 = 0) *dt[-1]
    r2 = np.append(r2, r2[-1] +dr2)

    # h = dt[-1]
    # k1 = dr2dt(r2[-1], m1, m2, rho_DM_at_r2 = 0)
    # k2 = dr2dt(r2[-1] +h *k1/2, m1, m2, rho_DM_at_r2 = 0)
    # k3 = dr2dt(r2[-1] +h *k2/2, m1, m2, rho_DM_at_r2 = 0)
    # k4 = dr2dt(r2[-1] +h *k3, m1, m2, rho_DM_at_r2 = 0)

    # r2 = np.append(r2, r2[-1] +h/6 *( k1 +2*k2 +2*k3 +k4 ))

    # Break the propagation if we reach the Innermost Stable Circular Orbit
    if r2[-1] < Risco: break

  r2 = r2[:-1]
  T = getPeriodFromDistance(r2, m1 +m2)
  t = np.cumsum(dt)

  return t, r2, 1/T