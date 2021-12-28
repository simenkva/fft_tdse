# fft_tdse

This is a simple toolbox for the solution of the time-dependent Schrödinger
equation using split-step Fourier method. The toolbox is written in a fairly
general fashion, for multiple dimensions, and assumes that the Hamiltonian is
on the form

$$ H(t) = T + V + E(t)*X, $$

where $T$ is diagonal in the Fourier basis, and where $V$ is diagonal in the
spatial basis. $E(t)$ is a time-dependent scalar function, and $X$ is again
diagonal in the spatial basis. Currently, $X$ is hard coded, and this will
change in the next version.
