from .simulator import Simulator
from icecream import ic
from .fft_tdse import GroundStateComputer, FourierHamiltonian, FourierWavefunction
from scipy.sparse.linalg import LinearOperator
import numpy as np
from scipy.sparse.linalg import eigsh, eigs, expm_multiply


class MatrixSimulator(Simulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_eigenstates(self, n_states=None):
        """Computes the eigenstates of the Hamiltonian.

        Stores the eigenvalues in self.E_vec and the eigenvectors in self.U_mat.

        Args:
            n_states (int): number of eigenstates to compute

        Returns:

        """

        self.n_states = n_states

        # get the appropriate grid
        if hasattr(self, "grid"):
            grid = self.grid
        else:
            raise ValueError("No grid has been set.")

        xx = grid.xx

        # create a hamiltonian object
        if not hasattr(self, "ham"):
            self.setup_hamiltonian()
        ham = self.ham

        # create a ground state computer object
        gs = GroundStateComputer(ham)
        # if guess is None:
        #     grid_shape = grid.xx[0].shape
        #     guess = np.random.rand(*grid_shape) - 0.5
        # gs.setInitialGuess(guess)

        # compute the dimension of the matrix
        n = np.prod(grid.ng)
        ic(n)

        # compute appropriate shift
        sigma = np.min(ham.V)

        # create a linear operator
        H_op = LinearOperator(
            (n, n), matvec=lambda x: gs.Apsi_vec(x, sigma=0), dtype=complex
        )

        # use arnoldi method to compute the lowest eigenvalues
        # using shoft and invert mode.
        # next, store the results in the class
        ic("Computing n_states eigenvalues of the Hamiltonian ... ")
        E, U = eigsh(
            H_op, k=n_states, which="LM", sigma=sigma, return_eigenvectors=True
        )

        self.E_vec = E
        self.U_mat = U
        self.H_mat = np.diag(self.E_vec)  # store the Hamiltonian matrix as well

        # # normalize eigenvectors to be real
        # normalize = True
        # if normalize:
        #     ic('normalizing eigenvectors ...')
        #     for i in range(n_states):
        #         U[:,i] = U[:,i].real
        #         U[:,i] /= np.linalg.norm(U[:,i])

        ic(self.E_vec)
        ic(self.E_vec.shape, self.U_mat.shape)

        # check that states are orthogonal
        ic("orthogonality check:")
        ic(
            np.linalg.norm(
                np.einsum("ki,kj->ij", self.U_mat.conj(), self.U_mat) - np.eye(n_states)
            )
        )

        # compute matrix representation of laser potential
        if hasattr(self, "laser_potential_fun"):
            D = self.laser_potential_fun(*grid.xx)
            temp = np.einsum("k,ki->ki", D, self.U_mat)
            ic(temp.shape, self.U_mat.shape)
            self.D_mat = np.einsum("ki,kj->ij", self.U_mat.conj(), temp)
            ic("Laser potential matrix:")
            ic(self.D_mat)

    def time_step(self):
        """Time steps the wavefunction by one time step.

        Uses the matrix representation of the Hamiltonian to time step the wavefunction.

        Args:

        Returns:

        """

        # get the time step
        dt = self.dt

        # get the wavefunction
        psi_vec = self.psi_vec

        # get the hamiltonian matrix
        H = self.H_mat

        # get the laser potential matrix
        if hasattr(self, "D_mat"):
            D = self.D_mat
        else:
            D = 0

        # compute the time step using sparse matrix exponential

        psi_vec = expm_multiply(-0.5j * dt * H, psi_vec)
        psi_vec = expm_multiply(
            -1j * dt * self.charge * D * self.laser_pulse_fun(self.t + 0.5 * dt),
            psi_vec,
        )
        psi_vec = expm_multiply(-0.5j * dt * H, psi_vec)

        # store the wavefunction
        self.psi_vec = psi_vec

    def prepare(self):
        # the super class prepare does some unnecessary things,
        # but ok.

        # make sure that the super class does not do inverse iterations ...
        def dummy_initial_condition(x):
            ic("This intial condition will not really be used.")
            return 0 * x

        self.set_initial_condition(dummy_initial_condition)
        super().prepare()

        if not hasattr(self, "psi_vec"):
            # set up ground state as initial condition/
            self.psi_vec = np.zeros((self.n_states,), dtype=complex)
            self.psi_vec[0] = 1
            ic(self.psi_vec)
        else:
            ic("using provided initial state vector")
            ic(self.psi_vec)

    def compute_psi(self):
        """Computes the wavefunction in real space."""

        self.psi = (
            np.einsum("ki,i->k", self.U_mat, self.psi_vec) / self.grid.dtau**0.5
        )

    def set_psi_vec_from_psi(self, psi):
        """Sets psi_vec by projecting a given wavefunction."""

        self.psi_vec = np.einsum("ki,k->i", self.U_mat.conj(), psi)
        self.psi_vec /= np.linalg.norm(self.psi_vec)
