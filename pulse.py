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

class Henriksen:
    def __init__(self, E0, omega, t_p = 0, sigma=None, shift_field=False):
        self.E0 = E0
        self.omega = omega
        self.t_p = t_p
        if sigma is None:
            self.sigma = np.pi/(omega * 2 * np.log(2))
        else:
            self.sigma = sigma
        self.shift_field = shift_field

    def _envelope(self, t):
        dt = t - self.t_p
        return np.exp(-dt**2/self.sigma**2)

    def __call__(self, t):
        if self.shift_field:
            dt = t - self.t_p
            return self.E0 * np.cos(self.omega*dt) * self._envelope(t)
        else:
            return self.E0 * np.cos(self.omega*t) * self._envelope(t)

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
    def __init__(self, E0, omega, envelope, shift_field=True):
        self.E0 = E0
        self.omega = omega
        self.envelope = envelope
        self.shift_field = shift_field

    def __call__(self, t):
        if self.shift_field:
            dt = t - self.get_center()
            return self.E0 * np.cos(self.omega*dt) * self.envelope(t)
        else:
            return self.E0 * np.cos(self.omega*t) * self.envelope(t)

    def get_duration(self):
        return self.envelope.get_param()[2]

    def get_center(self):
        return self.envelope.get_param()[1]


def setup_pulse(n=2, shift_field=False, verbose=False):
    '''Returns a callable laser pulse object constructed according to
    Machholm and Henriksen, Phys. Rev. Lett. 87, 193001 (2001), with
    field strength 4e9 V/m which can be generated by optical rectification
    from organic salt crystals, see 
    Fulop, Tzortzakis, and Kampfrath, Adv. Opt. Mater. 8, 1900681 (2020).
    The envelope is trigonometric (controlled by n)

    shift_field: controls whether to use cos(omega * t) [shift_field=False]
                 or cos(omega * (t - t_p)) [shift_field=True]

    '''
    E0_ = 4e9 # V/m
    omega_ = 36 # cm-1
    sigma = 279 # fs  --> corresponds to envelope exp(-(t-t_p)**2 / sigma**2) used by Machholm and Henriksen

    conv = EnergyConversions()

    E0 = E0_ / const.physical_constants['atomic unit of electric field'][0]
    omega = conv.cm12Eh(omega_)
    tau = np.sqrt(2*np.log(2)) * sigma/T0
    t_m = 0

    G = Envelope(n=n, t_m=0, FWHM=tau)
    Tm = G.get_param()[2]
    G.set_center(Tm/2)
    E = Laser(E0, omega, G, shift_field=shift_field)

    if verbose:
        print("Laser pulse:")
        print(f"    E0 = {E0_} V/m = {E0} a.u.")
        print(f"    omega = {omega_} cm-1 = {omega} a.u.")
        print(f"    sigma = {sigma} fs")
        print(f"    tau = {tau} a.u. = {tau*T0} fs")
        print(f"    omega = {omega} a.u. = {conv.Eh2cm1(omega)} cm-1")
        print(f"    omega = {omega} a.u. = {conv.Eh2nm(omega)} nm = {conv.Eh2nm(omega)*1e-6} mm")
        print(f"    omega = {omega} a.u. = {conv.Eh2Hz(omega)*1e-12} THz")
        f2f_duration = E.get_duration()
        print(f"    Pulse foot-to-foot duration: {f2f_duration} a.u. = {f2f_duration*T0} fs = {f2f_duration*T0*1e-3} ps")
        print(f"    Pulse peak intensity: {E0} a.u. = {intensity(E0)*1e-12} TW/cm2")

    return E


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    shift_field = False

    n = 10
    E = setup_pulse(n=n, shift_field=shift_field, verbose=True)
    t_min = 0
    t_max = 2*E.get_duration()
    ngrid = int(10*(t_max - t_min)) + 1
    t = np.linspace(t_min, t_max, num=ngrid)

    E0 = E.E0
    omega = E.omega
    t_p = E.get_center()
    sigma = 279/T0
    prl = Henriksen(E0, omega, t_p=t_p, sigma=sigma, shift_field=shift_field)
    print(f"Henriksen sigma = {prl.sigma} a.u. = {prl.sigma*T0} fs")

    fig = plt.figure()
    plt.plot(t*T0, E(t), label=f"laser (n={n})")
    plt.plot(t*T0, prl(t), label=f"Henriksen")
    plt.legend()
    plt.xlabel(r"$t$/fs")
    plt.ylabel(r"$E(t)$/a.u.")

    fig2 = plt.figure()
    plt.plot(t*T0, intensity(E(t))*1e-9, label=f"intensity(n={n})")
    plt.plot(t*T0, intensity(prl(t))*1e-9, label=f"intensity(Henriksen)")
    plt.legend()
    plt.xlabel(r"$t$/fs")
    plt.ylabel(r"$I(t)$/(GW/cm$^2$)")

    fig3 = plt.figure()
    plt.title("Envelopes")
    plt.plot(t*T0, E.envelope(t), label=f"laser (n={n})")
    plt.plot(t*T0, prl._envelope(t), label=f"Henriksen")
    plt.legend()
    plt.xlabel(r"$t$/fs")

    plt.show()
    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)