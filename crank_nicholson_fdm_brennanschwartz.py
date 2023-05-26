import numpy as np




class AmericanOption_crank_brennan:
    def __init__(self, T, S, K, sigma, r, q, option_type='put'):
        # Initialize the AmericanOption_crank_brennan class with the given parameters
        self.T = T  # Time to expiration
        self.S = S  # Current asset price
        self.K = K  # Strike price
        self.sigma = sigma  # Volatility
        self.r = r  # Risk-free interest rate
        self.q = q  # Dividend yield
        self.option_type = option_type  # Option type (default is 'put')

        # Initialize parameters for the Crank-Nicolson scheme and Brennan-Schwartz algorithm
        self.tol = 0.001  # Tolerance for convergence in Brennan-Schwartz algorithm
        self.omega = 1.2  # Relaxation parameter for SOR in Brennan-Schwartz algorithm

        # Set the maximum value for the asset price
        self.S_max = 10.5 * self.K + 100

        # Set the number of steps for time and asset price discretization
        self.N = 250  # Number of time steps
        self.M = 100  # Number of asset price steps
        self.dt = self.T / self.N  # Time step size
        self.ds = self.S_max / self.M  # Asset price step size

        # Create arrays for indices in the asset price and time dimensions
        self.I = np.arange(0, self.M + 1)  # Asset price indices
        self.J = np.arange(0, self.N + 1)  # Time indices

        # Initialize arrays for old and new option values
        self.old_val = np.zeros(self.M - 1)  # Old option values at asset price nodes
        self.new_val = np.zeros(self.M - 1)  # New option values at asset price nodes

        # Calculate the payoff of the option at expiration
        self.payoff = self.get_payoff()

        # Set the old option values at expiration as the initial layer
        self.old_layer = self.payoff

        # Calculate the boundary values at expiration
        self.bound_val = self.K * np.exp(-self.r * (self.N - self.J) * self.dt)

        # Calculate coefficients for the tridiagonal matrix in the implicit scheme
        self.alpha = 0.25 * self.dt * (self.sigma ** 2 * (self.I ** 2) - (self.r - self.q) * self.I)
        self.alpha = self.alpha[1:]
        self.beta = -self.dt * 0.5 * (self.sigma ** 2 * (self.I ** 2) + self.r)
        self.beta = self.beta[1:]
        self.gamma = 0.25 * self.dt * (self.sigma ** 2 * (self.I ** 2) + self.r * self.I)
        self.gamma = self.gamma[1:]

        # Construct the tridiagonal matrix for the implicit scheme
        self.M2 = np.diag(1 + self.beta[:self.M - 1]) + np.diag(self.alpha[1:self.M - 1], k=-1) + np.diag(
            self.gamma[:self.M - 2], k=1)

        # Initialize the right-hand side vector for the linear system
        self.b = np.zeros(self.M - 1)


    def get_payoff(self, q=0):
        S = self.I[1:self.M] * self.ds
        if self.option_type == 'put':
            payoff = np.maximum(self.K - S, 0)
        elif self.option_type == 'call':
            payoff = np.maximum(S - self.K, 0)
        else:
            raise ValueError("Invalid option type")

        # adjust the payoff for the dividend yield
        if q != 0:
            t = (self.N - self.J) * self.dt
            d = np.exp(-q * t)
            payoff = payoff * d

        return payoff

    def brennan_schwartz_algorithm(self):
        for j in range(self.N - 1, -1, -1): # Iterate backwards in time
            self.b[0] = self.alpha[0] * (self.bound_val[j] + self.bound_val[j + 1]) 
            rhs = self.M2 @ self.old_layer + self.b # Calculate the right-hand side vector
            self.old_val = self.old_layer # Set the old option values to the old layer
            error = 1000000
            while error > self.tol: # Iterate until the error is below the tolerance
                self.new_val[0] = self.old_val[0] + (self.omega / (1 - self.beta[0])) * (
                        rhs[0] - (1 - self.beta[0]) * self.old_val[0] + self.gamma[0] * self.old_val[1])
                for k in range(1, self.M - 2): # Update the option values
                    self.new_val[k] = self.old_val[k] + (self.omega / (1 - self.beta[k])) * (
                        rhs[k] - (1 - self.beta[k]) * self.old_val[k] + self.alpha[k] * self.new_val[k - 1] +
                        self.gamma[k] * self.old_val[k + 1]) # Update the option values
                self.new_val[self.M - 2] = self.old_val[self.M - 2] + (
                    self.omega / (1 - self.beta[self.M - 2])) * (
                                                      rhs[self.M - 2] - (1 - self.beta[self.M - 2]) *
                                                      self.old_val[self.M - 2] + self.alpha[self.M - 2] *
                                                      self.new_val[self.M - 3]) # Update the option values 
                self.new_val = np.maximum(self.new_val, self.payoff) # Apply the pointwise maximum
                error = np.linalg.norm(self.new_val - self.old_val) # Calculate the error
                self.old_val = self.new_val.copy() # Update the old option values 
            self.old_layer = self.new_val.copy() # Update the old layer
        return self.old_layer[int(self.S / self.ds) - 1] # Return the option value at the initial asset price


# Define the parameters
T = 1
S = 100
K = 100
sigma = 0.6
r = 0.02
q = 0.01




