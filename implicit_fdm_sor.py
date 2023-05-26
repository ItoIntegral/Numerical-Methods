import numpy as np


def european_option_price_implicit_sor(K, T, sigma, r, q, S, option_type):
    """
    Calculates the price of a European call or put option using the Implicit FDM and Successive Over-Relaxation (SOR) method.

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
    f[:, N] = np.maximum(K - (I * ds), 0)
    f[0, :] = K * np.exp(-r * (T - J * dt))
    f[M, :] = 0

    # Set up the tridiagonal matrix
    alpha = 0.25 * dt * ((sigma ** 2 * (I ** 2)) - (r - q) * I)
    beta = -dt * 0.5 * ((sigma ** 2 * (I ** 2)) + r)
    gamma = 0.25 * dt * ((sigma ** 2 * (I ** 2)) + (r - q) * I)
    M1 = np.diag(1 - beta[1:M]) + np.diag(-alpha[2:M], k=-1) + np.diag(-gamma[1:M - 1], k=1)
    M2 = np.diag(1 + beta[1:M]) + np.diag(alpha[2:M], k=-1) + np.diag(gamma[1:M - 1], k=1)

    # Set up SOR parameters
    omega = 1.2  # relaxation parameter
    tolerance = 1e-6  # convergence tolerance
    max_iterations = 1000  # maximum number of iterations

    # Solve for the option price at each time step using SOR
    for j in range(N - 1, -1, -1):
        l = np.zeros(M - 1)
        l[0] = alpha[1] * (f[0, j] + f[0, j + 1])
        l[-1] = gamma[M - 1] * (f[M, j] + f[M, j + 1])

        # SOR iteration loop
        for _ in range(max_iterations):
            prev_f = np.copy(f[1:M, j])  # store previous values of f for convergence check
            f[1:M, j] = np.linalg.solve(M1, M2 @ f[1:M, j + 1] + l)
            f[1:M, j] = omega * f[1:M, j] + (1 - omega) * prev_f  # relaxation

            # Check for convergence
            if np.linalg.norm(f[1:M, j] - prev_f, np.inf) < tolerance:
                break

    # Return the option price
    if option_type == 'call':
        return f[int(S / ds), 0]
    elif option_type == 'put':
        return f[int(S / ds), 0] + K * np.exp(-r * T) - S
    else:
        raise ValueError('option_type must be either "call" or "put"')



#print both the call and put option prices
    #define the parameters
K = 100
T = 1
sigma = 0.6
r = 0.02
q = 0.01
S = 100

