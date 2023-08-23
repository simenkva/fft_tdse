import numpy as np
import scipy.constants as const

Eh = const.physical_constants['atomic unit of energy'][0]
a0 = const.physical_constants['atomic unit of length'][0]
me = const.m_e
hbar = const.hbar

# masses in a.u.
#M_Li = 7.01600342665 * 1822.888486209
M_Li = 12786.393 # mass (in a.u.) from Tung, Pavanello, and Adamowicz, JCP 134, 064117 (2011)
m_p = const.m_p/const.m_e

# reduced proton mass
m = m_p*M_Li/(m_p + M_Li)

print(f"Reduced mass: m={m} a.u.")

# equilibrium distance
Re = 3.015
Re_SI = Re*a0

# moment of inertia
Ie = m*Re**2
Ie_SI = m*me*Re_SI**2

# rotational constant
Be = 1/(4*np.pi*Ie)
Be_SI = hbar/(4*np.pi*Ie_SI)
Be_cm1 = Be_SI/(const.c * 1e2)


# rotational period
tau = np.pi/Be
tau_SI = np.pi/Be_SI

print(f"Equilibrium distance: {Re} a.u. = {Re_SI} m = {Re_SI*1e10} A")
print(f"Moment of inertia:    {Ie} a.u. = {Ie_SI} kg m2")
print(f"Rotational constant:  {Be} a.u. = {Be_SI} Hz = {Be_SI*1e-12} THz = {Be_cm1} cm-1")
print(f"Rotational period:    {tau} a.u. = {tau_SI} s = {tau_SI*1e12} ps")
