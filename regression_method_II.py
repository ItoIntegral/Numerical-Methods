import numpy as np
from scipy.stats import norm


def regression_method_II_american_call(S, K, mu, r, delta, sigma, T, t0):
    # Calculate the number of time steps and paths
    M = int(T - t0 / 0.00125)
    N = 500

    # Calculate the time step size
    dt = (T - t0) / M

    # Initialize the matrix to store the stock prices at each time step
    S_values = np.zeros((M + 1, N))

    # Set the initial stock price
    S_values[0, :] = S

    # Generate stock price paths using geometric Brownian motion
    for i in range(M):
        S_values[i + 1, :] = S_values[i, :] * np.exp(
            (r - delta - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal(size=N))

    # Create an array of time points from t0 to T
    tau = np.linspace(t0, T, M + 1)

    # Initialize the option value array at the final time step
    g = np.maximum(S_values[-1] - K, 0)

    # Perform backward induction to calculate option values at each time step
    for i in range(M - 1, 0, -1):
        # Find the indices of paths where the stock price is greater than the strike price
        in_the_money = np.where(S_values[i] > K)[0]

        # Select the stock prices and option values for the in-the-money paths
        X = S_values[i][in_the_money]
        Y = np.exp(-r * (tau[-1] - tau[i]) * 4 * dt) * g[in_the_money]

        # Perform polynomial regression to estimate the continuation values
        A = np.vstack([np.ones_like(X), X, X ** 2, X ** 3]).T
        coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]

        # Calculate the estimated continuation values using the regression coefficients
        C = np.zeros(N)
        for j in range(len(C)):
            C[j] = coeffs[0] + coeffs[1] * S_values[i, j] + coeffs[2] * S_values[i, j] ** 2 + coeffs[3] * S_values[
                i, j] ** 3

        # Determine the optimal exercise decision by comparing the immediate exercise payoff with the estimated continuation values
        exercise = np.maximum(S_values[i] - K, 0)
        g[in_the_money] = np.maximum(exercise[in_the_money], C[in_the_money])

    # Calculate the discounted option value at time t0
    discount_factor = np.exp(-r * (tau[-1] - t0))
    C0 = discount_factor * np.mean(g)

    # Calculate the Black-Scholes option value for comparison
    d1 = (np.log(S / K) + (r - delta + 0.5 * sigma ** 2) * (T - t0)) / (sigma * np.sqrt(T - t0))
    d2 = d1 - sigma * np.sqrt(T - t0)
    BS_call = S * np.exp(-delta * (T - t0)) * norm.cdf(d1) - K * np.exp(-r * (T - t0)) * norm.cdf(d2)

    # Calculate the error between the estimated option value and the Black-Scholes option value
    error = np.abs(C0 - BS_call)

    return C0, error


def regression_method_II_american_put(S, K, mu, r, delta, sigma, T, t0):
    # Calculate the number of time steps and paths
    M = int(T - t0 / 0.00125)
    N = 500

    # Calculate the time step size
    dt = (T - t0) / M

    # Initialize the matrix to store the stock prices at each time step
    S_values = np.zeros((M + 1, N))

    # Set the initial stock price
    S_values[0, :] = S

    # Generate stock price paths using geometric Brownian motion
    for i in range(M):
        S_values[i + 1, :] = S_values[i, :] * np.exp(
            (r - delta - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal(size=N))

    # Create an array of time points from t0 to T
    tau = np.linspace(t0, T, M + 1)

    # Initialize the option value array at the final time step
    g = np.maximum(K - S_values[-1], 0)

    # Perform backward induction to calculate option values at each time step
    for i in range(M - 1, 0, -1):
        # Find the indices of paths where the stock price is less than the strike price
        in_the_money = np.where(S_values[i] < K)[0]

        # Select the stock prices and option values for the in-the-money paths
        X = S_values[i][in_the_money]
        Y = np.exp(-r * (tau[-1] - tau[i]) * 4 * dt) * g[in_the_money]

        # Perform polynomial regression to estimate the continuation values
        A = np.vstack([np.ones_like(X), X, X ** 2, X ** 3]).T
        coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]

        # Calculate the estimated continuation values using the regression coefficients
        C = np.zeros(N)
        for j in range(len(C)):
            C[j] = coeffs[0] + coeffs[1] * S_values[i, j] + coeffs[2] * S_values[i, j] ** 2 + coeffs[3] * S_values[
                i, j] ** 3

        # Determine the optimal exercise decision by comparing the immediate exercise payoff with the estimated continuation values
        exercise = np.maximum(K - S_values[i], 0)
        g[in_the_money] = np.maximum(exercise[in_the_money], C[in_the_money])

    # Calculate the discounted option value at time t0
    discount_factor = np.exp(-r * (tau[-1] - t0))
    C0 = discount_factor * np.mean(g)

    # Calculate the Black-Scholes option value for comparison
    d1 = (np.log(S / K) + (r - delta + 0.5 * sigma ** 2) * (T - t0)) / (sigma * np.sqrt(T - t0))
    d2 = d1 - sigma * np.sqrt(T - t0)
    BS_put = K * np.exp(-r * (T - t0)) * norm.cdf(-d2) - S * np.exp(-delta * (T - t0)) * norm.cdf(-d1)

    # Calculate the error between the estimated option value and the Black-Scholes option value
    error = np.abs(C0 - BS_put)

    return C0, error


# define the parameters
S = 100
K = 100
mu = 0.05
r = 0.02
delta = 0.01
sigma = 0.6
T = 1
t0 = 0


