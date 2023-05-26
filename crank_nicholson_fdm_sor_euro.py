import numpy as np



class EuropeanOption_crank_sor:
    def __init__(self, T, S, K, sigma, r, q, option_type='put'):
        # Initialize the EuropeanOption_crank_sor class with the given parameters
        self.T = T  # Time to expiration
        self.S = S  # Current asset price
        self.K = K  # Strike price
        self.sigma = sigma  # Volatility
        self.r = r  # Risk-free interest rate
        self.q = q  # Dividend yield
        self.option_type = option_type  # Option type (default is 'put')

        # Initialize parameters for the Crank-Nicolson scheme and SOR method
        self.tol = 0.001  # Tolerance for SOR convergence
        self.omega = 1.2  # Relaxation parameter for SOR

        # Set the maximum value for the asset price
        self.S_max = 2.314 * self.K + 100

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
        # Calculate the payoff of the option at expiration

        # Calculate the asset prices at the interior nodes
        S = self.I[1:self.M] * self.ds
        # Determine the payoff based on the option type

        if self.option_type == 'put':
            payoff = np.maximum(self.K - S, 0)  # Payoff for a put option
        elif self.option_type == 'call':
            payoff = np.maximum(S - self.K, 0)  # Payoff for a call option
        else:
            raise ValueError("Invalid option type")

        # Adjust the payoff if there is a dividend yield (q)
        if q != 0:
            t = (self.N - self.J) * self.dt  # Time to expiration for each time step
            d = np.exp(-q * t)  # Discount factor for the dividend yield
            payoff = payoff * d  # Adjust the payoff by multiplying with the discount factor

        return payoff

    def get_price(self):
        for j in range(self.N - 1, -1, -1):
            if self.option_type == 'call': # changed boundary condition for call option
                self.bound_val[j] = (self.S_max - self.K) * np.exp(
                    -self.r * (self.N - j) * self.dt)  # changed boundary condition for call option
            elif self.option_type == 'put': # existing boundary condition for put option
                self.bound_val[j] = self.K * np.exp(
                    -self.r * (self.N - j) * self.dt)  # existing boundary condition for put option

            self.b[0] = self.alpha[0] * (self.bound_val[j] + self.bound_val[j + 1])
            rhs = self.M2 @ self.old_layer + self.b
            self.old_val = self.old_layer # changed the old value
            error = 1000000 # initialize error
            while error > self.tol: # changed the while loop
                self.new_val[0] = self.old_val[0] + self.omega * (rhs[0] - (1 + self.beta[0]) * self.old_val[0] +
                                                                  self.gamma[0] * self.old_val[1]) / (

                                          1 + self.beta[0]) # changed the first node
                for k in range(1, self.M - 2): # changed the last node
                    self.new_val[k] = self.old_val[k] + (self.omega / (1 - self.beta[k])) * (
                            rhs[k] - (1 - self.beta[k]) * self.old_val[k] + self.alpha[k] * self.new_val[k - 1] +
                            self.gamma[k] * self.old_val[k + 1]) # changed the last node
                self.new_val[self.M - 2] = self.old_val[self.M - 2] + (
                        self.omega / (1 - self.beta[self.M - 2])) * (
                                                   rhs[self.M - 2] - (1 - self.beta[self.M - 2]) *
                                                   self.old_val[self.M - 2] + self.alpha[self.M - 2] *
                                                   self.new_val[self.M - 3]) # changed the last node

                error = np.linalg.norm(self.new_val - self.old_val) 
                self.old_val = self.new_val.copy() # Update the old option values with the new values
            self.old_layer = self.new_val.copy() # Update the old option values with the new values
        return self.old_layer[int(self.S / self.ds) - 1] # return the option value at the initial asset price node


# define the parameters
T = 1
S = 100
K = 100
sigma = 0.6
r = 0.02
q = 0.01


