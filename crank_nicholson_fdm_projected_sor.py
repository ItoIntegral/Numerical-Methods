

import numpy as np


class AmericanOption_crank_projectedsor:
    def __init__(self, T, S, K, sigma, r, q, option_type='put'):
        # Parameters of the option
        self.T = T  # Time to expiration
        self.S = S  # Initial stock price
        self.K = K  # Strike price
        self.sigma = sigma  # Volatility of the underlying asset
        self.r = r  # Risk-free rate
        self.q = q  # Dividend yield
        self.option_type = option_type  # Type of option ('call' or 'put')

        # Parameters for the numerical method
        self.tol = 0.001  # Tolerance for the iteration
        self.omega = 1.2  # Over-relaxation parameter

        # Grid parameters
        self.S_max = 2 * self.K + 100  # Maximum stock price considered
        self.N = 250  # Number of time steps
        self.M = 100  # Number of price steps
        self.dt = self.T / self.N  # Time step size
        self.ds = self.S_max / self.M  # Price step size

        # Indices for the grid
        self.I = np.arange(0, self.M + 1)  # Index array for the stock price
        self.J = np.arange(0, self.N + 1)  # Index array for the time

        # Arrays to store the old and new values during iteration
        self.old_val = np.zeros(self.M - 1)
        self.new_val = np.zeros(self.M - 1)

        # Calculate the payoff of the option
        self.payoff = self.get_payoff()
        self.old_layer = self.payoff

        # Calculate the boundary values
        self.bound_val = self.K * np.exp(-self.r * (self.N - self.J) * self.dt)

        # Coefficients for the finite-difference scheme
        self.alpha = 0.25 * self.dt * (self.sigma ** 2 * (self.I ** 2) - (self.r - self.q) * self.I)
        self.alpha = self.alpha[1:]
        self.beta = -self.dt * 0.5 * (self.sigma ** 2 * (self.I ** 2) + self.r)
        self.beta = self.beta[1:]
        self.gamma = 0.25 * self.dt * (self.sigma ** 2 * (self.I ** 2) + self.r * self.I)
        self.gamma = self.gamma[1:]

        # Construct the tridiagonal matrix for the Crank-Nicolson method
        self.M2 = np.diag(1 + self.beta[:self.M - 1]) + np.diag(self.alpha[1:self.M - 1], k=-1) + np.diag(
            self.gamma[:self.M - 2], k=1)
        self.b = np.zeros(self.M - 1)  # Right-hand side of the linear system


    def get_payoff(self, q=0):
        S = self.I[1:self.M] * self.ds # stock price
        if self.option_type == 'put': # payoff of put option
            payoff = np.maximum(self.K - S, 0) 
        elif self.option_type == 'call': # payoff of call option
            payoff = np.maximum(S - self.K, 0)
        else:
            raise ValueError("Invalid option type")

        # adjust the payoff for the dividend yield
        if q != 0: 
            t = (self.N - self.J) * self.dt
            d = np.exp(-q * t)
            payoff = payoff * d

        return payoff 

    def get_price(self):
        for j in range(self.N - 1, -1, -1): # backward induction
            self.b[0] = self.alpha[0] * (self.bound_val[j] + self.bound_val[j + 1]) 
            rhs = self.M2 @ self.old_layer + self.b # right hand side
            self.old_val = self.old_layer # old layer
            error = 1000000 
            while error > self.tol: # SOR iteration
                self.new_val[0] = self.old_val[0] + (self.omega / (1 - self.beta[0])) * (
                            rhs[0] - (1 - self.beta[0]) * self.old_val[0] + self.gamma[0] * self.old_val[1])
                for k in range(1, self.M - 2):
                    self.new_val[k] = self.old_val[k] + (self.omega / (1 - self.beta[k])) * (
                            rhs[k] - (1 - self.beta[k]) * self.old_val[k] + self.alpha[k] * self.new_val[k - 1] +
                            self.gamma[k] * self.old_val[k + 1])
                self.new_val[self.M - 2] = self.old_val[self.M - 2] + (
                            self.omega / (1 - self.beta[self.M - 2])) * (
                                                              rhs[self.M - 2] - (1 - self.beta[self.M - 2]) *
                                                              self.old_val[self.M - 2] + self.alpha[self.M - 2] *
                                                              self.new_val[self.M - 3])
                self.new_val = np.maximum(self.new_val, self.payoff) # check for early exercise
                error = np.linalg.norm(self.new_val - self.old_val) # error
                self.old_val = self.new_val.copy() # update the old value
            self.old_layer = self.new_val.copy() # update the old layer
        return self.old_layer[int(self.S / self.ds) - 1] # return the price of the option



#define the parameters
T = 1
S = 100
K = 100
sigma = 0.6
r = 0.02
q = 0.01

