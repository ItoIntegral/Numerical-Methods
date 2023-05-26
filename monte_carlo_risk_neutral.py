import numpy as np


def calculate_european_option_price(S, K, r, delta, sigma, T, option_type, num_paths, num_steps):
    # Calculate the price of a European option using Monte Carlo simulation

    dt = T / num_steps  # Time step size

    # Initialize matrix to store asset price paths
    S_paths = np.zeros((num_paths, num_steps + 1))
    S_paths[:, 0] = S  # Set initial asset price for all paths

    # Generate asset price paths using geometric Brownian motion
    for i in range(num_paths):
        for j in range(num_steps):
            z = np.random.standard_normal()  # Generate a random number from a standard normal distribution
            S_paths[i, j + 1] = S_paths[i, j] * np.exp((r - delta - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
            # Update asset price using the geometric Brownian motion equation

    # Calculate payoffs based on the option type
    if option_type == 'call':
        payoffs = np.maximum(S_paths[:, -1] - K, 0)  # Payoff for a call option
    elif option_type == 'put':
        payoffs = np.maximum(K - S_paths[:, -1], 0)  # Payoff for a put option
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    discount_factor = np.exp(-r * T)  # Discount factor for the risk-free interest rate
    option_price = discount_factor * np.mean(payoffs)  # Calculate the option price as the discounted average of payoffs

    return option_price


# Parameters
S = 100  # Price of the underlying asset today
K = 100  # Strike price
r = 0.02  # Risk-free interest rate
delta = 0.01  # Dividend yield
sigma = 0.6  # Volatility
T = 1  # Maturity date

num_paths = 500  # Number of sample paths
num_steps = 800  # Number of steps (M)

# Calculate option prices
call_price = calculate_european_option_price(S, K, r, delta, sigma, T, 'call', num_paths, num_steps)
put_price = calculate_european_option_price(S, K, r, delta, sigma, T, 'put', num_paths, num_steps)


