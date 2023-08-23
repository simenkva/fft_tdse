import numpy as np
from lih_potential import LiH_Potential, LiH_Dipole
import scipy.constants as const

#
# Morse potential
#
def morse(r, a = 1.0, D = 1.0, r_e = 1.0):
    return D * (1.0 - np.exp(-a * (r-r_e)))**2 - D

#
# Smoothed Coulomb potential
#
def smooth_coulomb(r, a = 0.1):
    return (r**2 + a**2)**(-0.5)


#
# LiH potential
#
def get_lih_potentials(verbose=False):
    debye = 1e-21 / const.c  # definition in C m
    debye /= const.e * const.physical_constants['atomic unit of length'][0]
    eV = const.physical_constants['atomic unit of energy'][0]/const.e
    pot = LiH_Potential()
    dip = LiH_Dipole()
    if verbose:
        print(f"LiH potential curve: {pot.get_source()}")
        print(f"LiH dipole curve:    {dip.get_source()}")
        Re = pot.get_Re()
        print(f"    Re = {Re} bohr    Ve = {pot(Re)*1e3} mEh = {pot(Re)*eV} eV")
        print(f"    De = {dip(Re)} a.u. = {dip(Re)/debye} debye")
    return pot, dip


if __name__ == '__main__':
    V, D = get_lih_potentials(verbose=True)
