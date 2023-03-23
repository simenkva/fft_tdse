import numpy as np
import scipy.constants as const

T0 = const.physical_constants['atomic unit of time'][0] * 1e15

class EnergyConversions:
    def __init__(self):
        self.Eh = const.physical_constants['atomic unit of energy'][0]
        self.eV = self.Eh/const.e
        self.cm1 = self.Eh/(const.h*const.c*1e2)

    def Eh2eV(self, E):
        return E*self.eV

    def eV2Eh(self, E):
        return E/self.eV

    def Eh2cm1(self, E):
        return E*self.cm1

    def cm12Eh(self, E):
        return E/self.cm1

    def Eh2nm(self, E):
        return (1e9*const.h*const.c/E)/self.Eh

    def nm2Eh(self, lamda):
        return (1e9*const.h*const.c/lamda)/self.Eh

    def Eh2Hz(self, E):
        return E*self.Eh/const.h

    def Hz2Eh(self, E):
        return E*const.h/self.Eh

def intensity(E0):
    '''Field strength E0 in a.u.
       Returns the intensity in W/cm^2'''
    E2 = field_strength(E0)**2
    I = 0.5*const.epsilon_0*const.c*E2
    return I*1.e-4

def field_strength(E0):
    '''Field strength E0 in a.u.
       Returns the electric field strength in V/m'''
    return E0*const.physical_constants['atomic unit of electric field'][0]

class Envelope:

    def __init__(self, n=2, t_m=0, FWHM=1):
        self.n = n
        self.t_m = t_m
        self.T_m = np.pi * FWHM / (2 * np.arccos(2**(-1/(2 * n))))

    def _heaviside(self, dt):
        return np.heaviside(dt + self.T_m/2, 1) * np.heaviside(self.T_m/2 - dt, 1)

    def __call__(self, t):
        dt = t - self.t_m
        cos = np.cos(np.pi*dt/self.T_m)**self.n
        return cos * self._heaviside(dt)

    def set_center(self, t_m):
        self.t_m = t_m

    def get_param(self):
        return self.n, self.t_m, self.T_m


class Laser:
    def __init__(self, E0, omega, envelope):
        self.E0 = E0 
        self.omega = omega
        self.envelope = envelope

    def __call__(self, t):
        dt = t - self.envelope.t_m
        return self.E0 * np.cos(self.omega*dt) * self.envelope(t)

    def get_duration(self):
        return self.envelope.get_param()[2]

    def get_center(self):
        return self.envelope.get_param()[1]


def setup_pulse(n=2, verbose=False):
    '''Returns a callable laser pulse object constructed according to
    Machholm and Henriksen, Phys. Rev. Lett. 87, 193001 (2001)
    The envelope is trigonometric (controlled by n)
    '''
    E0_ = 1.5e9 # V/m
    omega_ = 36 # cm-1
    duration = 450 # fs

    conv = EnergyConversions()

    E0 = E0_ / const.physical_constants['atomic unit of electric field'][0]
    omega = conv.cm12Eh(omega_)
    FWHM = duration/T0
    t_m = 0

    G = Envelope(n=n, t_m=0, FWHM=FWHM)
    Tm = G.get_param()[2]
    G.set_center(Tm/2)
    E = Laser(E0, omega, G)

    if verbose:
        print("Laser pulse:")
        print(f"    E0 = {E0_} V/m = {E0} a.u.")
        print(f"    omega = {omega_} cm-1 = {omega} a.u.")
        print(f"    duration = {duration} fs = {FWHM} a.u.")
        print(f"    omega = {omega} a.u. = {conv.Eh2cm1(omega)} cm-1")
        print(f"    omega = {omega} a.u. = {conv.Eh2nm(omega)} nm = {conv.Eh2nm(omega)*1e-6} mm")
        print(f"    omega = {omega} a.u. = {conv.Eh2Hz(omega)*1e-12} THz")
        print(f"    Pulse duration: {FWHM} a.u. = {duration} fs = {duration*1e-3} ps")
        f2f_duration = E.get_duration()
        print(f"    Pulse foot-to-foot duration: {f2f_duration} a.u. = {f2f_duration*T0} fs = {f2f_duration*T0*1e-3} ps")
        print(f"    Pulse peak intensity: {E0} a.u. = {intensity(E0)} W/cm2")

    return E


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n = 2
    E = setup_pulse(n=n, verbose=True)
    t_min = 0
    t_max = 2*E.get_duration()
    ngrid = int(10*(t_max - t_min)) + 1
    t = np.linspace(t_min, t_max, num=ngrid)

    fig = plt.figure()
    plt.plot(t*T0, E(t), label=f"laser (n={n})")
    plt.legend()
    plt.xlabel(r"$t$/fs")
    plt.ylabel(r"$E(t)$/a.u.")

    fig2 = plt.figure()
    plt.plot(t*T0, intensity(E(t)), label=f"intensity(n={n})")
    plt.legend()
    plt.xlabel(r"$t$/fs")
    plt.ylabel(r"$I(t)$/(W/cm$^2$)")

    plt.show()
    plt.close(fig)
    plt.close(fig2)
