# HySE

_A simulation toolbox for the time-dependent Schrödinger equation dveloped for ease-of-use, aiming for physics and chemistry didactical use, but also reaseach. The toolbox is developed by Simen Kvaal at the Hylleraas Centre for Quantum Molecular Sciences at the University of Oslo, Norway._

<img src="Hyse.jpg" alt="HySE" title="HySE" width="320" />


## About

This is a simulation toolbox for simulating the time-dependent Schrödinger equation,

$$ i \partial_t \Psi(x,t) = H(t) \Psi(x,t), \quad \Psi(0) \text{ given}, $$

where $x \in \mathbb{R}^n$, using the split-step Fourier method. The basic functionality is very general and supports Hamiltonians of the form

$$ H(t) = T(k) + V(x) + U(x,t), $$

where $T(k)$ is any operator diagonal in momentum space, and $V(x)$ and $U(x,t)$ are diagonal in position space.
Additionally, $U(x,t)$ depends explicitly on time.


The contents of this repository can be summarized as follows:
  * A package with a toolbox for simulating the time-dependent Schroedinger equation
    on a grid using the split-step FFT method.
  * A simulator class for easily setting up simulations in 1d, 2d, and 3d.
  * Animator classes for easily producing visually appealing animations using matplotlib and FFMPEG.
  * A collection of demo notebooks for 1d and 2d simulations:  <./demo_notebooks> .
    * You can run the demo notebooks and produce the animations using <./demo_notebooks/run_notebooks.py>.
  * TO BE UPDATED: A collection of configurable sample simulation scripts in 1d, 2d, and 3d

## Installation


Clone the repository and run `pip install .`. This will install the `fft_tdse` toolbox.
The scripts are not installed. Probably you would like to edit them and the config files
to suit your own needs. Thus, copy them to your desired location and edit away.

To compile one of the demo notebooks, move to the `demo_notebooks` folder and run
```
python run_notebooks.sh -p Hydrogen model in 1d.ipynb
```
This particular command will produce a movie `atom_1d.mp4`

## Contributing

To contribute, please follow these guidelines:

1. Fork the repository and create a new branch for your contribution.
2. Make your changes and ensure they align with the project's goals and coding conventions.
3. Test your changes thoroughly to ensure they do not introduce any regressions.
4. Submit a pull request with a clear description of your changes and their purpose.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Please see the [LICENSE](LICENSE) file for more information.

