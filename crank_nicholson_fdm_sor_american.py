import numpy as np



class AmericanOption_crank_regularsor:
    def __init__(self, T, S_0, K, sigma, r, q, option_type='put'):
        self.T = T
        self.S_0 = S_0
        self.K = K
        self.sigma = sigma
        self.r = r
        self.q = q
        self.option_type = option_type

        self.tol = 0.001 # tolerance
        self.omega = 1.2 # relaxation parameter

        self.S_max = 2 * self.K + 100 # the upper bound of the stock price

        self.N = 250 # number of time steps
        self.M = 100 # number of stock price steps
        self.dt = self.T / self.N # time step
        self.ds = self.S_max / self.M # stock price step

        self.I = np.arange(0, self.M + 1) # stock price grid
        self.J = np.arange(0, self.N + 1) # time grid

        self.old_val = np.zeros(self.M - 1) # old layer
        self.new_val = np.zeros(self.M - 1) # new layer

        self.payoff = self.get_payoff() # payoff
        self.old_layer = self.payoff # old layer
        self.bound_val = self.K * np.exp(-self.r * (self.N - self.J) * self.dt) # boundary value

        self.alpha = 0.25 * self.dt * (self.sigma ** 2 * (self.I ** 2) - (self.r - self.q) * self.I) 
        self.alpha = self.alpha[1:] 
        self.beta = -self.dt * 0.5 * (self.sigma ** 2 * (self.I ** 2) + self.r)
        self.beta = self.beta[1:]
        self.gamma = 0.25 * self.dt * (self.sigma ** 2 * (self.I ** 2) + (self.r - self.q) * self.I)
        self.gamma = self.gamma[1:]

        self.M2 = np.diag(1 + self.beta[:self.M - 1]) + np.diag(self.alpha[1:self.M - 1], k=-1) + np.diag(
            self.gamma[:self.M - 2], k=1) # matrix M2
        self.b = np.zeros(self.M - 1) # vector b

    def get_payoff(self):
        if self.option_type == 'put': # put option
            return np.maximum(self.K - self.I[1:self.M] * self.ds, 0) 
        elif self.option_type == 'call': # call option
            return np.maximum(self.I[1:self.M] * self.ds - self.K, 0)

    def get_price(self):
        for j in range(self.N - 1, -1, -1):
            self.b[0] = self.alpha[0] * (self.bound_val[j] + self.bound_val[j + 1]) 
            rhs = self.M2 @ self.old_layer + self.b # get the right hand side
            self.old_val = self.old_layer # update the old layer
            error = 1000000
            while error > self.tol:
                self.new_val[0] = np.maximum(self.payoff[0], self.old_val[0] + (self.omega / (1 - self.beta[0])) * (
                            rhs[0] - (1 - self.beta[0]) * self.old_val[0] + self.gamma[0] * self.old_val[1]))
                for k in range(1, self.M - 2):
                    self.new_val[k] = np.maximum(self.payoff[k], self.old_val[k] + (self.omega / (1 - self.beta[k])) * (
                            rhs[k] - (1 - self.beta[k]) * self.old_val[k]                             + self.alpha[k] * self.new_val[k - 1] + self.gamma[k] * self.old_val[k + 1]))
                self.new_val[self.M - 2] = np.maximum(self.payoff[self.M - 2], self.old_val[self.M - 2] + (
                            self.omega / (1 - self.beta[self.M - 2])) * (
                                                              rhs[self.M - 2] - (1 - self.beta[self.M - 2]) *
                                                              self.old_val[self.M - 2] + self.alpha[self.M - 2] *
                                                              self.new_val[self.M - 3]))
                error = np.linalg.norm(self.new_val - self.old_val) 
                self.old_val = self.new_val.copy() # update the old layer
            self.old_layer = self.new_val.copy() # update the old layer
        return self.old_layer[int(self.S_0 / self.ds) - 1] # return the price of the option

#define the parameters
T = 1
S_0 = 100
K = 100
sigma = 0.6
r = 0.02
q = 0.01


