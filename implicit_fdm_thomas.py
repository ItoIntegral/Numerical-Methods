import numpy as np


def european_option_price_implicit_thomas(K, T, sigma, r, q, S, option_type):
    """
    Calculates the price of a European call or put option using the Implicit FDM and Thomas algorithm.

    Parameters:
    K (float): strike price
    T (float): time to maturity (in years)
    sigma (float): volatility
    r (float): risk-free interest rate
    q (float): dividend yield
    S (float): current price of the underlying asset
    option_type (str): 'call' for call option or 'put' for put option

    Returns:
    float: price of the option
    """
    # Set up parameters for the FDM
    S_max = 2 * K  # set the maximum price for the grid
    N = 500  # number of time steps
    M = 50  # number of price steps
    dt = T / N
    ds = S_max / M

    # Set up the mesh of approximation
    f = np.zeros((M + 1, N + 1))
    I = np.arange(0, M + 1)
    J = np.arange(0, N + 1)

    # Set the boundary and final conditions
    if option_type == 'call':
        f[:, N] = np.maximum((I * ds) - K, 0)
        f[0, :] = 0
        f[M, :] = S_max - K * np.exp(-r * (T - J * dt)) # For large S, the call option payoff approaches S_max - K*exp(-r*(T-t))
    elif option_type == 'put':
        f[:, N] = np.maximum(K - (I * ds), 0)
        f[0, :] = K * np.exp(-r * (T - J * dt)) # For small S, the put option payoff approaches K*exp(-r*(T-t))
        f[M, :] = 0
    else:
        raise ValueError('option_type must be either "call" or "put"')


    # Set up the tridiagonal matrix
    alpha = 0.25 * dt * ((sigma ** 2 * (I ** 2)) - (r - q) * I)
    beta = -dt * 0.5 * ((sigma ** 2 * (I ** 2)) + r)
    gamma = 0.25 * dt * ((sigma ** 2 * (I ** 2)) + (r - q) * I)
    M1 = np.diag(1 - beta[1:M]) + np.diag(-alpha[2:M], k=-1) + np.diag(-gamma[1:M - 1], k=1)
    M2 = np.diag(1 + beta[1:M]) + np.diag(alpha[2:M], k=-1) + np.diag(gamma[1:M - 1], k=1)

    # Solve for the option price at each time step
    for j in range(N - 1, -1, -1):
        l = np.zeros(M - 1)
        l[0] = alpha[1] * (f[0, j] + f[0, j + 1])
        l[-1] = gamma[M - 1] * (f[M, j] + f[M, j + 1])
        f[1:M, j] = np.linalg.solve(M1, M2 @ f[1:M, j + 1] + l)

    # Interpolate to get the option price at S
    idown = int(np.floor(S / ds))
    iup = int(np.ceil(S / ds))
    if idown == iup:
        price = f[idown, 0]
    else:
        price = f[idown, 0] + ((iup - (S / ds)) / (iup - idown)) * (f[iup, 0] - f[idown, 0])

    # Adjust for call or put option
    if option_type == 'call':
        return price
    elif option_type == 'put':
        return price - S + K
    else:
        raise ValueError('option_type must be "call" or "put"')

    # define the parameters


K = 100
T = 1
sigma = 0.6
r = 0.02
q = 0.01
S = 100

