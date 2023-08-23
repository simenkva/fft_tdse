import numpy as np


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
# Henon-Heiles potential
#

# TODO:

# define it here
# edit simulation scripts accordingly.