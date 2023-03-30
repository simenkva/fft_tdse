import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pulse import setup_pulse
import subprocess

class LiH_Data:
    def __init__(self, prepend=False, append=False, keep4_25=True):
        self.source = "Partridge et al., J. Chem. Phys. 75 2299 (1981)"
        if keep4_25:
            comp_raw = np.array([[1.750, -7.779181, -1.9182],
                                 [2.000, -7.836250, -1.9955],
                                 [2.250, -7.872266, -2.0403],
                                 [2.500, -7.894965, -2.0525],
                                 [2.750, -7.909183, -2.0409],
                                 [2.875, -7.914133, -2.0277],
                                 [3.000, -7.918024, -2.0099],
                                 [3.125, -7.921076, -1.9881],
                                 [3.250, -7.923461, -1.9610],
                                 [3.500, -7.926749, -1.8936],
                                 [3.750, -7.928686, -1.8076],
                                 [3.875, -7.929326, -1.7576],
                                 [4.000, -7.929807, -1.7025],
                                 [4.125, -7.930260, -1.6422],
                                 [4.250, -7.930422, -1.5770],
                                 [4.375, -7.930606, -1.5078],
                                 [4.500, -7.930730, -1.4309],
                                 [4.750, -7.930843, -1.2575],
                                 [4.900, -7.930840, -1.1396],
                                 [5.000, -7.930792, -1.0542],
                                 [5.500, -7.930356, -0.5403],
                                 [5.750, -7.929927, -0.2190],
                                 [6.000, -7.929314,  0.1495],
                                 [6.500, -7.927416,  1.0213],
                                 [7.000, -7.924533,  1.9896],
                                 [7.500, -7.920767,  2.8948],
                                 [8.000, -7.916461,  3.5979],
                                 [8.500, -7.911952,  4.0292],
                                 [9.000, -7.907657,  4.1951],
                                 [9.500, -7.903746,  4.0784],
                                 [10.00, -7.900372,  3.6761],
                                 [10.50, -7.897636,  3.0364],
                                 [11.00, -7.895576,  2.2929],
                                 [12.00, -7.893153,  1.0525],
                                 [13.50, -7.891901,  0.2677],
                                 [15.00, -7.891586,  0.0705],
                                 [17.50, -7.891443,  0.0125],
                                 [20.00, -7.891434,  0.0031],
                                 [22.50, -7.891428,  0.0004],
                                 [25.00, -7.891426,  0.000469]])
        else:
            comp_raw = np.array([[1.750, -7.779181, -1.9182],
                                 [2.000, -7.836250, -1.9955],
                                 [2.250, -7.872266, -2.0403],
                                 [2.500, -7.894965, -2.0525],
                                 [2.750, -7.909183, -2.0409],
                                 [2.875, -7.914133, -2.0277],
                                 [3.000, -7.918024, -2.0099],
                                 [3.125, -7.921076, -1.9881],
                                 [3.250, -7.923461, -1.9610],
                                 [3.500, -7.926749, -1.8936],
                                 [3.750, -7.928686, -1.8076],
                                 [3.875, -7.929326, -1.7576],
                                 [4.000, -7.929807, -1.7025],
                                 [4.125, -7.930260, -1.6422],
                                 [4.375, -7.930606, -1.5078],
                                 [4.500, -7.930730, -1.4309],
                                 [4.750, -7.930843, -1.2575],
                                 [4.900, -7.930840, -1.1396],
                                 [5.000, -7.930792, -1.0542],
                                 [5.500, -7.930356, -0.5403],
                                 [5.750, -7.929927, -0.2190],
                                 [6.000, -7.929314,  0.1495],
                                 [6.500, -7.927416,  1.0213],
                                 [7.000, -7.924533,  1.9896],
                                 [7.500, -7.920767,  2.8948],
                                 [8.000, -7.916461,  3.5979],
                                 [8.500, -7.911952,  4.0292],
                                 [9.000, -7.907657,  4.1951],
                                 [9.500, -7.903746,  4.0784],
                                 [10.00, -7.900372,  3.6761],
                                 [10.50, -7.897636,  3.0364],
                                 [11.00, -7.895576,  2.2929],
                                 [12.00, -7.893153,  1.0525],
                                 [13.50, -7.891901,  0.2677],
                                 [15.00, -7.891586,  0.0705],
                                 [17.50, -7.891443,  0.0125],
                                 [20.00, -7.891434,  0.0031],
                                 [22.50, -7.891428,  0.0004],
                                 [25.00, -7.891426,  0.000469]])
        if prepend:
            x = np.linspace(0, 1.75, num=int(4*1.75 + 1), endpoint=False)
            e = 2.1125 * np.exp(-1.5051 * x) - 7.9308514
            d = -1.91826
            gen_low = np.array([[x[i], e[i], d] for i in range(x.shape[0])])
            tmp = np.concatenate((gen_low, comp_raw), axis=0)
        else:
            tmp = comp_raw
        if append:
            x = np.linspace(27.5, 50.0, num=int(4*(50.0-27.5) + 1))
            e = -147.9/x**6 - 20700/x**8 - 7.89142584
            d = 0.000469
            gen_high = np.array([[x[i], e[i], d] for i in range(x.shape[0])])
            self.raw = np.concatenate((tmp, gen_high), axis=0)
        else:
            self.raw = tmp

    def get_source(self):
        return self.source

    def get_R(self):
        N = self.raw.shape[0]
        res = np.array([self.raw[i, 0] for i in range(N)])
        return res

    def get_E(self):
        N = self.raw.shape[0]
        res = np.array([self.raw[i, 1] for i in range(N)])
        return res

    def get_d(self):
        N = self.raw.shape[0]
        res = np.array([self.raw[i, 2] for i in range(N)])
        return res

class LiH_Potential:
    def __init__(self, zero_point='dissociation', fitted=False, keep4_25=True):
        if fitted:
            self.Re = 4.510915722494084
            self.De = 0.03956542973471824
            self.a = 0.7124920885811191
            self.b = 0.21876029990323365
            self.c = 0.060226371736622375
            self.d = -1.0084971503460642e-05
            self.e = -0.0017872616361749325
            self.f = -0.00015132930379472076
            self.g = 2.303765617402607e-05
            if zero_point.lower() == 'dissociation':
                self.h = 0
            else:
                self.h = -7.891391513847319
            self.pot = self._fitted
        else:
            self.zero_point = zero_point
            data = LiH_Data(prepend=False, append=False, keep4_25=keep4_25)
            x = data.get_R()
            e = data.get_E()
            self.cs = CubicSpline(x, e)
            self.pot = self._interp

    def __call__(self, R):
        return self.pot(R)

    def _fitted(self, R):
        dR = R - self.Re
        return self.h - self.De*(1 + self.a*dR + self.b*dR**2
                        + self.c*dR**3 + self.d*dR**4 + self.e*dR**5
                        + self.f*dR**6 + self.g*dR**7)*np.exp(-self.a*dR)

    def _low(self, R):
        return 2.1125 * np.exp(-1.5051 * R) - 7.9308514

    def _high(self, R):
        return -147.9/R**6 - 20700/R**8 - 7.89142584

    def _interp(self, R):
        cond = [R < 1.75, ((1.75 <= R) & (R <= 25.0)), 25.0 < R]
        fun = [self._low, self.cs, self._high]
        V = np.piecewise(R, cond, fun)
        if self.zero_point == 'dissociation':
            return V - self._high(100)
        else:
            return V


class LiH_Dipole:
    def __init__(self, fitted=False, keep4_25=True):
        if fitted:
            self.Re = 8.80826679063919
            self.De = -4.221782190714436
            self.a = 0.060504494997313585
            self.b = -0.10177939733508684
            self.c = 0.003203916018615962
            self.d = 0.002427370245358651
            self.e = 3.4994557127168364e-05
            self.f = -5.36284925127309e-05
            self.g = 3.15445760842076e-06
            self.h = -1.6897876241894067e-05
            self.dip = self._fitted
        else:
            data = LiH_Data(prepend=False, append=False, keep4_25=keep4_25)
            x = data.get_R()
            d = data.get_d()
            self.cs = CubicSpline(x, d)
            self.dip = self._interp

    def __call__(self, x, y=None, direction='x'):
        if y is None:
            return self.dip(x)
        else:
            r = np.sqrt(x**2 + y**2)
            d = self.dip(r)
            theta = np.arctan2(y, x)
            if direction.lower() == 'x':
                return d * np.cos(theta)
            elif direction.lower() == 'y':
                return d * np.sin(theta)
            else:
                raise ValueError(f"Illegal direction: {direction}")

    def _fitted(self, R):
        dR = R - self.Re
        return self.h - self.De*(1 + self.a*dR + self.b*dR**2
                        + self.c*dR**3 + self.d*dR**4 + self.e*dR**5
                        + self.f*dR**6 + self.g*dR**7)*np.exp(-self.a*dR**2)

    def _low(self, R):
        return np.full(R.shape[0], -1.91826)

    def _high(self, R):
        return np.full(R.shape[0], 0.000469)

    def _interp(self, R):
        cond = [R < 1.75, ((1.75 <= R) & (R <= 22.5)), 22.5 < R]
        fun = [self._low, self.cs, self._high]
        return np.piecewise(R, cond, fun)


def visualize_pot(t0, t1, n, animate=False, folder='out', max_show=11):
    V = LiH_Potential(fitted=False)
    D = LiH_Dipole(fitted=False)
    E = setup_pulse()
    Veff = lambda xx, yy, tt: V(np.sqrt(xx**2 + yy**2)) - D(xx, yy) * E(t)

    xmin = -15
    xmax = 15
    ng = 20*(xmax - xmin) + 1
    x = np.linspace(-15, 15, num=ng)
    y = np.linspace(-15, 15, num=ng)
    X, Y = np.meshgrid(x, y)

    vmin = -50
    vmax = 50
    scale = 1e3 # potential unit will be mEh

    if animate: 
        animation_file = f'{folder}/Veff_animation.mp4'
        framelist_file = f'{folder}/Veff_framelist.txt' 
        f = open(framelist_file, 'w')

    t_range = np.linspace(t0, t1, num=n)
    for i, t in enumerate(t_range):
        Z = Veff(X, Y, t) * scale
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        my_cmap = plt.get_cmap('jet')
        surf = ax.plot_surface(X, Y, Z, cmap=my_cmap, vmin=vmin, vmax=vmax, edgecolor=None, linewidth=0, antialiased=False)
        fig.colorbar(surf, ax=ax, shrink=0.7, aspect=7)
        ax.set_title(f"t={t:.6f} E={E(t):.6f}")
        ax.set_zlim(vmin, vmax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

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
    import scipy.constants as const

    ANIMATE = False
    KEEP4_25 = True
    UNIT = "cm-1"

    if UNIT.lower() == "au":
        unit = 1
        unit_label = "Hartree"
    elif UNIT.lower() == "ev":
        unit = const.physical_constants['atomic unit of energy'][0]/const.e
        unit_label = "eV"
    elif UNIT.lower() == "cm-1":
        unit = const.physical_constants['atomic unit of energy'][0]
        unit /= const.h*const.c*1e2
        unit *= 1e-3
        unit_label = r"$10^3$ cm$^{-1}$"
    else:
        raise ValueError(f"Illegal UNIT={UNIT}")

    Rmin = 1
    Rmax = 25
    ngrid = 100*(Rmax - Rmin) + 1
    R = np.linspace(Rmin, Rmax, num=ngrid)
    data = LiH_Data(prepend=False, append=False, keep4_25=KEEP4_25)

    V = LiH_Potential(fitted=False, keep4_25=KEEP4_25)
    Vfit = LiH_Potential(fitted=True)
    Rref = data.get_R()
    Vref = data.get_E()
    Vref -= Vref[-1]

    distances = np.linspace(3.000, 4.830, num=1000001)
    energies = np.array([V(r) for r in distances])
    ind = np.argmin(energies)
    print(f"Potential minimum: {energies[ind]} at {distances[ind]}")

    fig1 = plt.figure()
    plt.title("Potential")
    plt.plot(R, V(R)*unit, label="Interp")
    plt.plot(R, Vfit(R)*unit, label="Fitted")
    plt.plot(Rref, Vref*unit, 'o', label="Ref")
    plt.xlim(Rmin, Rmax)
    plt.ylim(-0.05*unit, 0.06*unit)
    plt.xlabel(r"$R$/bohr")
    plt.ylabel(fr"$V$/{unit_label}")
    plt.legend()

    D = LiH_Dipole(fitted=False, keep4_25=KEEP4_25)
    Dfit = LiH_Dipole(fitted=True)
    Dref = data.get_d()

    fig2 = plt.figure()
    plt.title("Dipole")
    plt.plot(R, D(R), label="Interp")
    plt.plot(R, Dfit(R), label="Fitted")
    plt.plot(Rref, Dref, 'o', label="Ref")
    plt.xlim(Rmin, Rmax)
    plt.xlabel(r"$R$/bohr")
    plt.ylabel(r"$d$/a.u.")
    plt.legend()
    plt.show()

    plt.close(fig1)
    plt.close(fig2)

    if ANIMATE:
        visualize_pot(0, 81500, 1001, animate=True)
