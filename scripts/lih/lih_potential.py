import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import scipy.constants as const
from pulse import setup_pulse
import subprocess

DEBYE = 1e-21 / const.c  # definition in C m
DEBYE /= const.e * const.physical_constants['atomic unit of length'][0]

class SRPot:
    def __init__(self, R1, E2, E0, deltaR):
        self.R1 = R1
        self.G1 = (E2 - E0)/deltaR

    def a(self, b):
        return -self.G1*np.exp(b*self.R1)/b

    def __call__(self, R, b, c):
        return self.a(b) * np.exp(-b * R) + c

def parse_Vdata(fname, withAdiabaticCorrection=True):
    if withAdiabaticCorrection:
        indx = 13
    else:
        indx = 1
    with open(fname, 'r') as f:
        lines = f.readlines()
        R = np.array([float(line.split()[0]) for line in lines[6:]])
        V = np.array([float(line.split()[indx]) for line in lines[6:]])
    return R, V

def parse_Ddata(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
        R = np.array([float(line.split()[0]) for line in lines])
        D = np.array([float(line.split()[1]) for line in lines])
    return R, D

class LiH_Potential:
    def __init__(self, withAdiabaticCorrection=True, zero_point='dissociation'):
        self.source = "Tung, Pavanello, Adamowicz; J. Chem. Phys. 134, 064117 (2011). DOI:10.1063/1.3554211"
        if withAdiabaticCorrection:
              self.source + " (BO with adiabatic correction)"
        else:
              self.source + " (BO without adiabatic correction)"
        self.Rdata, self.Vdata = parse_Vdata("data/lih_potential_adamowicz.txt", withAdiabaticCorrection=withAdiabaticCorrection)
        self._setupSR(self.Rdata[:3], self.Vdata[:3])
        if zero_point == 'dissociation':
            self.Voo = self.Vdata[-1]
        else:
            self.Voo = 0
        self.cs = CubicSpline(self.Rdata, self.Vdata)

    def get_source(self):
        return self.source

    def get_R(self):
        return self.Rdata

    def get_E(self):
        return self.Vdata

    def get_Re(self, n=200001):
        Re_approx = self.Rdata[np.argmin(self.Vdata)]
        Rmin = Re_approx - 0.1
        Rmax = Re_approx + 0.1
        R = np.linspace(Rmin, Rmax, num=n)
        V = self.__call__(R)
        return R[np.argmin(V)]

    def _setupSR(self, Rdata, Vdata):
        model = SRPot(Rdata[1], Vdata[2], Vdata[0], Rdata[2]-Rdata[0])
        pini = np.array([2.0, -8.1])
        popt, pcov = curve_fit(model, Rdata, Vdata, p0=pini)
        self.b = popt[0]
        self.a = model.a(self.b)
        self.c = popt[1]

    def _short_range(self, R):
        return self.a * np.exp(-self.b*R) + self.c

    def _long_range(self, R):
        return np.full(R.shape[0], self.Voo)

    def __call__(self, R):
        cond = [R < self.Rdata[1], ((self.Rdata[1] <= R) & (R <= self.Rdata[-1])), self.Rdata[-1] < R]
        fun = [self._short_range, self.cs, self._long_range]
        V = np.piecewise(R, cond, fun)
        return V - self.Voo


class LiH_Dipole:
    def __init__(self, unit='a.u.'):
        self.source = "Diniz, Kirnosov, Alijah, Mohallem, Adamowicz; J. Mol. Spectrosc. 322, 22-28 (2016). DOI: 10.1016/j.jms.2016.03.001"
        self.Rdata, self.Ddata = parse_Ddata("data/lih_dipole_adamowicz.txt")
        if unit.lower() == 'a.u.' or unit.lower() == 'au' or unit.lower() == 'ea0':
            self.Ddata *= DEBYE # convert from Debye to a.u.
            self.source += ' (unit: a.u.)'
        elif unit.lower() == 'debye':
            self.source += ' (unit: debye)'
        else:
            raise ValueError(f"Unknown dipole unit: {unit}")
        x_extra = np.array([(self.Rdata[-1] + 10.*(i+1)) for i in range(2)])
        y_extra = np.array([0., 0.])
        xdata = np.concatenate((self.Rdata, x_extra))
        ydata = np.concatenate((self.Ddata, y_extra))
        self.cs = CubicSpline(xdata, ydata)
        self.Rlong = x_extra[0]

    def get_source(self):
        return self.source

    def get_R(self):
        return self.Rdata

    def get_D(self):
        return self.Ddata

    def _short_range(self, R):
        return np.full(R.shape[0], self.Ddata[0])

    def _long_range(self, R):
        return np.full(R.shape[0], 0.)

    def dipole(self, R):
        cond = [R < self.Rdata[0], ((self.Rdata[0] <= R) & (R <= self.Rlong)), self.Rlong < R]
        fun = [self._short_range, self.cs, self._long_range]
        return np.piecewise(R, cond, fun)

    def __call__(self, x, y=None, direction='x'):
        if y is None:
            return self.dipole(x)
        else:
            r = np.sqrt(x**2 + y**2)
            d = self.dipole(r)
            theta = np.arctan2(y, x)
            if direction.lower() == 'x':
                return d * np.cos(theta)
            elif direction.lower() == 'y':
                return d * np.sin(theta)
            else:
                raise ValueError(f"Illegal direction: {direction}")


def visualize_pot(t0, t1, n, animate=False, folder='out', max_show=11):
    V = LiH_Potential()
    D = LiH_Dipole()
    E = setup_pulse()
    Veff = lambda xx, yy, tt: V(np.sqrt(xx**2 + yy**2)) - D(xx, yy) * E(tt)

    xmin = -10
    xmax = 10
    ng = 20*(xmax - xmin) + 1
    x = np.linspace(-15, 15, num=ng)
    y = np.linspace(-15, 15, num=ng)
    X, Y = np.meshgrid(x, y)

    vmin = -110
    vmax = 110
    scale = 1e3 # potential unit will be mEh

    if animate: 
        animation_file = f'{folder}/Veff_animation.mp4'
        framelist_file = f'{folder}/Veff_framelist.txt' 
        f = open(framelist_file, 'w')

    t_range = np.linspace(t0, t1, num=n)
    cmap = plt.get_cmap('jet')
    for i, t in enumerate(t_range):
        Z = Veff(X, Y, t) * scale
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor=None, linewidth=0, antialiased=False)
        fig.colorbar(surf, ax=ax, shrink=0.7, aspect=7)
        ax.set_title(f"t={t:.6f} E={E(t):.6f}")
        ax.set_zlim(vmin, vmax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.elev = 20

        if animate:
            framename = f'Veff_{i:05d}.png'
            filename = f'{folder}/{framename}'
            plt.savefig(filename, dpi = 100)
            print(f"file {framename}", file=f)
            print(f"Veff frame {i:05d} generated")
        else:
            plt.show()
        plt.close(fig)

        if not animate and i+1 > max_show:
            print("Max number of figs reached. Stopping now.")
            break

    if animate:
        f.close()
        cmd = f'ffmpeg -y -r 30 -f concat -i {framelist_file} -vcodec libx264 -crf 25  -pix_fmt yuv420p {animation_file}'
        subprocess.run(cmd, shell=True)



if __name__ == '__main__':
    ANIMATE = True

    V = LiH_Potential()
    D = LiH_Dipole()
    Re = V.get_Re()
    Ve = V(Re)
    print(f"Re= {Re} bohr     V(Re) = {Ve*1e3} mEh    Dipole(Re) = {D(Re)} a.u. = {D(Re)/DEBYE} debye")
    R = np.linspace(0, 60, 6001)

    fig = plt.figure()
    plt.plot(V.get_R(), V.get_E(), 'o', color="black")
    plt.plot(R, V(R))
    plt.ylim(Ve-0.01, 0.1)

    fig1 = plt.figure()
    DMC = D(R)
    plt.plot(D.get_R(), D.get_D(), 'o', color="black")
    plt.plot(R, DMC)
    plt.ylim(DMC.min()-0.05, DMC.max()+0.05)
    

    plt.show()
    plt.close(fig)
    plt.close(fig1)


    if ANIMATE:
        visualize_pot(0, 81500, 1001, animate=True)
