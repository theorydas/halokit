from sympy.physics.quantum.cg import Wigner3j
from sympy.physics.quantum.spin import Rotation
from scipy import integrate
import scipy.special as sc
import mpmath as mp
import numpy as np
import math

# ======== Functions

def OmegaKepler(rs, R_star):
    """Just the Keplerian formula for Omega0, for the effective 2-body problem (so, (1+q)*rs should be used in reality)
    """
    
    return np.sqrt(rs/(2 *R_star**3))

def Rbound(n, l, r, rs, alpha):
    """wavefunction of bound states"""
    
    rbohr = rs / (2*alpha**2)
    return np.sqrt((4*math.factorial(n -l -1))/(rbohr**3 *n**4*math.factorial(n+l))) * np.exp(-r/(n*rbohr)) * (2*r/(n*rbohr))**l * sc.genlaguerre(n-l-1,2*l+1)(2*r/(n*rbohr))

def Runbound(k,l,r,rs,alpha): # wavefunction of unbound states
    mu = 2*alpha/rs
    eta = -mu*alpha / k
    rho = k*r
    return (2/r) * np.frompyfunc(mp.coulombf,3,1)(l,eta,rho)
    # return (2/r) * np.sin(rho) # this would be the approximation that does not take into account the deformation of unbound states due to the gravity of the central BH

def C(lprime,l_star,l,mprime,m_star,m): # angular integral
    return (-1)**abs(mprime+m_star) * np.sqrt(((2*lprime+1)*(2*l_star+1)*(2*l+1))/(4*np.pi)) * float(Wigner3j(lprime, 0, l_star, 0, l, 0).doit()) * float(Wigner3j(lprime, -mprime, l_star, -m_star, l, m).doit())

def K(lprime,l_star,l,k,n,R_star,rs,alpha): # radial integral

    def integrand1(r):
        return r**(2+l_star)*Runbound(k,lprime,r,rs,alpha)*Rbound(n,l,r,rs,alpha)
    
    def integrand2(r):
        return r**(2-l_star-1)*Runbound(k,lprime,r,rs,alpha)*Rbound(n,l,r,rs,alpha)
    
    return np.sqrt(rs) * (integrate.quad(integrand1,0,R_star)[0]/R_star**(l_star+1) + integrate.quad(integrand2,R_star,np.inf)[0]*R_star**l_star)

# ===== Rate Calculation

def Rate_Original(n, l, m, q, R_star, chi, rs, alpha, lprimeMAX, epsilon, getContributions = False):
  # returns the (dN/dt)/N, saving the partial sums for the maximum value of lf going from 0 to lfMAX. The mass of the cloud is replaced by M_c/M_BH
    Omega_star = OmegaKepler(rs *(1 +q), R_star)
    mu = 2 *alpha/rs
    E_n = - mu *alpha**2/(2 *n**2)
    
    partial_sum_rate = np.zeros(lprimeMAX +1)
    partial_sum_dEdt = np.zeros(lprimeMAX +1)
    partial_sum_dLdt = np.zeros(lprimeMAX +1)
    
    for lprime in range(0, lprimeMAX +1):
        
        # print(r"Computing r_0/r_S = {:.2f},".format(R_star/rs),"l' = {:.0f}".format(lprime))
        
        # This can be zero. So basically initilize the first lol.
        if not getContributions and lprime > 0:
            partial_sum_rate[lprime] = partial_sum_rate[lprime -1]
            partial_sum_dEdt[lprime] = partial_sum_dEdt[lprime -1]
            partial_sum_dLdt[lprime] = partial_sum_dLdt[lprime -1]
        
        for mprime in range(-lprime, lprime +1):

            if chi == 0:
                if m -mprime <= np.floor(E_n/Omega_star):
                    msecond_values = range(m -mprime, m -mprime +1)
                else:
                    msecond_values = range(m -mprime, m -mprime)
            elif chi == np.pi:
                if -m +mprime <= np.floor(E_n/Omega_star):
                    msecond_values = range(-m +mprime, -m +mprime+ 1)
                else:
                    msecond_values = range(-m +mprime, -m +mprime)
            else:
                msecond_values = range(-lprime-l, int(np.floor(E_n/Omega_star)) + 1)
            
            for msecond in msecond_values:
                
                Upsilon = 0
                k_msecond = np.sqrt(2*mu*(E_n-msecond*Omega_star))
                
                for l_star in [x for x in range(max(abs(msecond), abs(lprime -l)), lprime +l +1) if (x != 1)]:
                    
                    if (l_star>=abs(m-mprime) and (lprime+l_star+l)%2 == 0): # a simple check to exclude cases where C=0 due to selection rules
                        c = C(lprime, l_star, l,mprime, m-mprime, m)
                        D = Rotation.D(l_star, m -mprime, msecond, 0, chi, 0).doit()

                        if (abs(D) > 10**(-15) and abs(c) > 10**(-15)): # a simple check for speed-up
                            k = K(lprime,l_star,l,k_msecond,n,R_star,rs,alpha)
                            Y = sc.sph_harm(msecond,l_star,0,np.pi/2)
                            
                            Upsilon += k * c * D * Y / (2*l_star+1)
                
                # Save these 3 separately?
                partial_sum_rate[lprime] += -mu *abs(Upsilon)**2 /k_msecond
                partial_sum_dEdt[lprime] += msecond *Omega_star *mu *abs(Upsilon)**2 /k_msecond
                partial_sum_dLdt[lprime] += (m -mprime) *mu *abs(Upsilon)**2 /k_msecond

    # Multiply with physical units to normalize results.
    partial_sum_rate *= (4*np.pi * alpha * q)**2 / rs
    partial_sum_dEdt *= epsilon * (4*np.pi * alpha * q)**2 / (2*alpha)
    partial_sum_dLdt *= epsilon * (4*np.pi * alpha * q)**2 / (2*alpha)
    
    return np.array([partial_sum_rate, partial_sum_dEdt, partial_sum_dLdt])

vec_Rate_Original = np.vectorize(Rate_Original, otypes=[float])

def RateComponent(n, l, m, q, R_star, chi, rs, alpha, epsilon, lprime):
  # returns the (dN/dt)/N, saving the partial sums for the maximum value of lf going from 0 to lfMAX. The mass of the cloud is replaced by M_c/M_BH
    Omega_star = OmegaKepler(rs *(1 +q), R_star)
    mu = 2 *alpha/rs
    E_n = - mu *alpha**2/(2 *n**2)
    
    partial_sum_rate = 0
    partial_sum_dEdt = 0
    partial_sum_dLdt = 0

    for mprime in range(-lprime, lprime +1):

        if chi == 0:
            if m -mprime <= np.floor(E_n/Omega_star):
                msecond_values = range(m -mprime, m -mprime +1)
            else:
                msecond_values = range(m -mprime, m -mprime)
        elif chi == np.pi:
            if -m +mprime <= np.floor(E_n/Omega_star):
                msecond_values = range(-m +mprime, -m +mprime+ 1)
            else:
                msecond_values = range(-m +mprime, -m +mprime)
        else:
            msecond_values = range(-lprime-l, int(np.floor(E_n/Omega_star)) + 1)
        
        for msecond in msecond_values:
            
            Upsilon = 0
            k_msecond = np.sqrt(2*mu*(E_n-msecond*Omega_star))
            
            for l_star in [x for x in range(max(abs(msecond), abs(lprime -l)), lprime +l +1) if (x != 1)]:
                
                if (l_star>=abs(m-mprime) and (lprime+l_star+l)%2 == 0): # a simple check to exclude cases where C=0 due to selection rules
                    c = C(lprime, l_star, l,mprime, m-mprime, m)
                    D = Rotation.D(l_star, m -mprime, msecond, 0, chi, 0).doit()

                    if (abs(D) > 10**(-15) and abs(c) > 10**(-15)): # a simple check for speed-up
                        k = K(lprime,l_star,l,k_msecond,n,R_star,rs,alpha)
                        Y = sc.sph_harm(msecond,l_star,0,np.pi/2)
                        
                        Upsilon += k * c * D * Y / (2*l_star +1)
                        # Redefine Upsilon outside sympy.
                        r, i = Upsilon.as_real_imag()
                        Upsilon = float(r) +1j *float(i)
            
            # Save these 3 separately?
            partial_sum_rate += -mu *abs(Upsilon)**2 /k_msecond
            partial_sum_dEdt += msecond *Omega_star *mu *abs(Upsilon)**2 /k_msecond
            partial_sum_dLdt += (m -mprime) *mu *abs(Upsilon)**2 /k_msecond
    
    # Multiply with physical units to normalize results.
    partial_sum_rate *= -(4*np.pi * alpha * q)**2 / rs
    partial_sum_dEdt *= -epsilon * (4*np.pi * alpha * q)**2 / (2*alpha)
    partial_sum_dLdt *= -epsilon * (4*np.pi * alpha * q)**2 / (2*alpha)
    
    return float(partial_sum_rate), float(partial_sum_dEdt), float(partial_sum_dLdt)

def Rate(n, l, m, q, R_star, chi, rs, alpha, lprimeMAX, epsilon):
    partial_sum_rate = []
    partial_sum_dEdt = []
    partial_sum_dLdt = []
    
    for lprime in range(0, lprimeMAX +1):
      rate, dEdt, dLdt = RateComponent(n, l, m, q, R_star, chi, rs, alpha, epsilon, lprime)

      partial_sum_rate.append(rate)
      partial_sum_dEdt.append(dEdt)
      partial_sum_dLdt.append(dLdt)
    
    return np.array([partial_sum_rate, partial_sum_dEdt, partial_sum_dLdt])

vec_Rate = np.vectorize(Rate, otypes=[float])

def getDiscontinuity(f, q, n, alpha):
  rr_disc = f**(2/3) *(1 +q)**(1/3) *2**(-1/3) *n**(4/3) *alpha**-2

  return rr_disc