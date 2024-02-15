import numpy as np
from icecream import ic
from scipy.linalg import expm



class ImagProp():
    """ Class for imaginary time propagation in a subspace.
    
    The user supplies an imaginary time propagator, and the class
    implements a method for propagating a guess for the subspace basis
    until convergence.
    
    """
    
    def __init__(self, n, propagator):
        """
        Args:
            n (int): Dimension of the full space.
            propagator (callable): Function that takes a vector of length n
                and returns the result of applying the imaginary time propagator
                to the vector.
        """
        self.n = n
        self.propagator = propagator
    
    
    def imag_prop(self, psi_guess, tol=1e-6, maxit=1000, verbose=True, print_every=10):
        """Propagate a guess for the subspace basis until convergence.
        
        Args:
            psi_guess (ndarray): Initial guess for the subspace basis. The
                shape of the array should be (n, n_vec), where n_vec is the
                number of vectors in the subspace.
            tol (float): Tolerance for the convergence criterion.
            maxit (int): Maximum number of iterations.
            verbose (bool): If True, print information about the convergence.
            print_every (int): Print information every print_every iterations if verbose=True.
            
        Returns:
            psi (ndarray): The converged subspace basis.
            error (float): The final error estimate. 
        """
        
        # Check the input.
        assert psi_guess.shape[0] == self.n
        n_vec = psi_guess.shape[1]
        if verbose:
            ic(n_vec)
            
 
        # QR decompose the guess for the subspace basis.
        psi, _ = np.linalg.qr(psi_guess)
        
        # Do iterations until convergence.
        for k in range(maxit):
            # Propagate all basis vectors
            psi_new = np.zeros_like(psi)
            for i in range(n_vec):
                psi_new[:,i] = self.propagator(psi[:,i])
                
            # Orthonormalize the new basis vectors.
            psi_new, _ = np.linalg.qr(psi_new)
            
            # Check for convergence.
            error = np.linalg.norm(psi_new - psi, ord='fro')
            psi = psi_new
            if verbose and k % print_every == 0:
                ic(k, error)
            if error < tol:
                break
            
        # Return the converged basis and the final error estimate.
        return psi, error
        
            

if __name__ == '__main__':
    
    n = 100
    H = np.diag(np.arange(n)) + np.random.rand(n, n)*.1
    H = (H + H.T)/2
    U_imag = expm(-0.1*H)
    def propagator(psi):
        return np.dot(U_imag, psi)
    
    E_exact, U_exact = np.linalg.eigh(H)
    
    n_vec = 1
    U = U_exact[:,:n_vec]
    psi_init = np.random.rand(n, n_vec)
    ip = ImagProp(n, propagator)
    U_approx, error = ip.imag_prop(psi_init, tol=1e-8)
    
    error = U_approx - U @ U.T @ U_approx
    ic(np.linalg.norm(error))
    
#    ic(np.linalg.norm(U_exact[:,:n_vec]-psi*np.inner(psi,U_exact[:,0])))
#    ic(psi[:4], U_exact[:,0][:4])
    