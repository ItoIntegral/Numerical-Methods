
import numpy as np
from scipy.stats import norm


def closed_form_european_call(S, K, mu, r, delta, sigma, T, t0):
    """
    Calculate the price of a European call option using the Black-Scholes formula.

    Parameters:
    S (float): the current price of the underlying asset
    K (float): the strike price of the option
    mu (float): the mean return of the asset
    r (float): the riskless interest rate
    delta (float): the dividend yield of the asset
    sigma (float): the volatility (standard deviation) of the asset
    T (float): the time to expiration of the option, in years
    t0 (float): the current time, in years

    Returns:
    float: the price of the European call option
    """
    d1 = (np.log(S / K) + (r - delta + 0.5 * sigma ** 2) * (T - t0)) / (sigma * np.sqrt(T - t0)) # d1
    d2 = d1 - sigma * np.sqrt(T - t0) # d2
    return S * np.exp(-delta * (T - t0)) * norm.cdf(d1) - K * np.exp(-r * (T - t0)) * norm.cdf(d2) # Black-Scholes formula


def closed_form_european_put(S, K, mu, r, delta, sigma, T, t0):
    """
    Calculate the price of a European put option using the Black-Scholes formula.

    Parameters:
    S (float): the current price of the underlying asset
    K (float): the strike price of the option
    mu (float): the mean return of the asset
    r (float): the riskless interest rate
    delta (float): the dividend yield of the asset
    sigma (float): the volatility (standard deviation) of the asset
    T (float): the time to expiration of the option, in years
    t0 (float): the current time, in years

    Returns:
    float: the price of the European put option
    """
    d1 = (np.log(S / K) + (r - delta + 0.5 * sigma ** 2) * (T - t0)) / (sigma * np.sqrt(T - t0)) # d1
    d2 = d1 - sigma * np.sqrt(T - t0) # d2
    return K * np.exp(-r * (T - t0)) * norm.cdf(-d2) - S * np.exp(-delta * (T - t0)) * norm.cdf(-d1) # Black-Scholes formula



# Set parameter values
S = 100
K = 100
mu = 0.05
r = 0.02
delta = 0.01
sigma = 0.6
T = 1
t0 = 0



