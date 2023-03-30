import numpy as np
import matplotlib.pyplot as plt
import h5py

folder = 'out'
casename = 'sim_2_0'

sim_name = f"{folder}/{casename}"
fname = f'{sim_name}.hdf5'

with h5py.File(fname,'r') as h5file:

    t = np.squeeze(np.asarray(h5file['parameters/time']))
    E = np.squeeze(np.asarray(h5file['/parameters/hamiltonian/E']))
    H = np.squeeze(np.asarray(h5file['/energy']))
    #theta = np.squeeze(np.asarray(h5file['/theta']))
    #theta2 = np.squeeze(np.asarray(h5file['/theta2']))
    cos = np.squeeze(np.asarray(h5file['/cos']))
    cos2 = np.squeeze(np.asarray(h5file['/cos2']))

    figs = []

    fig = plt.figure()
    plt.plot(t, E)
    plt.title("Electric field")
    plt.xlabel(r"$t$ (a.u.)")
    plt.ylabel(r"$E(t)$ (a.u.)")
    fig.savefig('out/electric_field.pdf', dpi=300)
    figs.append(fig)

    #fig1 = plt.figure()
    #plt.plot(t, theta/np.pi)
    #plt.xlabel(r"$t$ (a.u.)")
    #plt.ylabel(r"$\theta/\pi$")
    #figs.append(fig1)

    #fig2 = plt.figure()
    #plt.plot(t, theta2/(np.pi**2))
    #plt.xlabel(r"$t$ (a.u.)")
    #plt.ylabel(r"$\theta^2/\pi^2$")
    #figs.append(fig2)

    #fig3 = plt.figure()
    #delta_theta = np.sqrt(theta2 - theta**2)
    #plt.plot(t, delta_theta/np.pi)
    #plt.xlabel(r"$t$ (a.u.)")
    #plt.ylabel(r"$\Delta\theta/\pi$")
    #figs.append(fig3)

    fig4 = plt.figure()
    plt.plot(t, cos)
    plt.xlabel(r"$t$ (a.u.)")
    plt.ylabel(r"$\langle \psi \vert \cos\theta \vert \psi \rangle$")
    fig4.savefig('out/cos_theta.pdf', dpi=300)
    figs.append(fig4)

    fig5 = plt.figure()
    plt.plot(t, cos2)
    plt.xlabel(r"$t$ (a.u.)")
    plt.ylabel(r"$\langle \psi \vert \cos^2\theta \vert \psi \rangle$")
    fig5.savefig('out/cos2_theta.pdf', dpi=300)
    figs.append(fig5)

    fig6 = plt.figure()
    plt.plot(t, H)
    plt.title("Hamiltonian")
    plt.xlabel(r"$t$ (a.u.)")
    plt.ylabel(r"$\langle \psi \vert H(t) \vert \psi \rangle$ (a.u.)")
    fig6.savefig('out/hamiltonian.pdf', dpi=300)
    figs.append(fig6)

    plt.show()
    for f in figs:
        plt.close(f)
