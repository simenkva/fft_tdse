import numpy as np
import matplotlib.pyplot as plt
import hsluv
import colorcet

for key in colorcet.cm.keys():
    print(key)

cmap = colorcet.cm["cyclic_mygbm_30_95_c78"]
# cmap = colorcet.cm['cyclic_rygcbmr_50_90_c64_s25']


def hsluv_to_rgb(h, s, l):
    fun = lambda h, s, l: hsluv.hsluv_to_rgb((h, s, l))
    return np.vectorize(fun)(h, s, l)


def rgb_to_hsluv(r, g, b):
    fun = lambda r, g, b: hsluv.rgb_to_hsluv((r, g, b))
    return np.vectorize(fun)(r, g, b)


def vis(f, xmin, xmax, ymin, ymax, nx, ny):
    x, y = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
    bmp = np.zeros((nx, ny, 3))

    z = f(x, y)
    hue = np.angle(z, deg=True) / 360 + 0.5
    color = cmap(hue)

    for i in range(3):
        bmp[:, :, i] = color[:, :, i] * np.abs(z).clip(0, 1)

    # h, s, l =  rgb_to_hsluv(color[:,:,0], color[:,:,1], color[:,:,2])
    # l2 = l * np.abs(z).clip(0, 1)
    # bmp[:,:,0], bmp[:,:,1], bmp[:,:,2]  = hsluv_to_rgb(h, s, l2)

    print(bmp.max(), bmp.min())

    plt.imshow(bmp, cmap=cmap)
    plt.colorbar()
    plt.show()


print(hsluv_to_rgb([360, 0], [10, 20], [50, 10]))


vis(lambda x, y: x + 1j * y, -1, 1, -1, 1, 100, 100)
