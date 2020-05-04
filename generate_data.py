import numpy as np
import matplotlib.pyplot as plt

# This utilities are used in the main program to generate the learning data y and U
# that illustrate the use of the module siso_predictor. 
# U and y will be the input and output of a dynamic system representing a reactor. 
# see https://ieeexplore.ieee.org/document/8277242 for more details regarding this system. 
# to be short, it is a highly nonlinear system that is commonly used to explore the convergence 
# behavior of Economic Model Predictive Control frameworks.

# control bounds
umin = 0.049
umax = 0.449
# sampling period
tau = 0.02
# order of the moments retaied for the features
order = 2
# parameters of the random scenarios
nUvalues = 400
max_duration = 100


def xdot(x, u):

    # The true unknown system of ODEs that governs the dynamics to be
    # identified.

    return np.array([
        1 - 1e4 * x[0]**2 * np.exp(-1.0 / x[2]) -
        400 * x[0] * np.exp(-0.55 / x[2]) - x[0],
        1e4 * x[0]**2 * np.exp(-1.0 / x[2]) - x[1],
        u - x[2]
    ])


def f(x, u):

    # The One-Step ahead simulator (using the tau sampling period)

    k1 = xdot(x, u)
    k2 = xdot(x + 0.5 * tau * k1, u)
    k3 = xdot(x + 0.5 * tau * k2, u)
    k4 = xdot(x + tau * k3, u)
    xplus = x + tau * (k1 + 2 * (k2 + k3) + k4) / 6.0
    xplus[0] = np.max(np.array([0, xplus[0]]))
    return xplus


def generate_random_excitation(plot=False):

    # Generate the random excitation scenario to build the learning data
    # for the identification of the model via ML.

    Uvalues = umin + np.random.rand(nUvalues) * (umax - umin)
    nPeriods = np.random.randint(1, max_duration, nUvalues)
    U = None
    for u, N in zip(Uvalues, nPeriods):
        if U is None:
            U = np.ones(N) * u
        else:
            inter = np.ones(N) * u
            U = np.array([*U, *inter])

    nt = len(U)
    t = np.linspace(0, nt * tau, nt)
    if plot:
        plt.plot(t, U, 'k', linewidth=2)
        plt.xlim([t.min(), t.max()])
        plt.grid(True)

    return t, U


def simulate_OL(x0, U):

    # simulates the open-loop behavior with a given input scenario.

    X = np.zeros((len(U) + 1, len(x0)))
    X[0, :] = x0
    for i in range(len(U)):
        X[i + 1, :] = f(X[i, :], U[i])

    return X
