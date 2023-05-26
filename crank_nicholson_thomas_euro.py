import numpy as np

class EuropeanOption_thomas:
    def __init__(self, T, S, K, sigma, r, q, option_type='put'):
        # T: time to maturity
        # S: spot price
        # K: strike price
        # sigma: volatility
        # r: risk-free rate
        # q: dividend yield
        # option_type: 'call' or 'put'
        
        
        self.T = T 
        self.S = S
        self.K = K
        self.sigma = sigma
        self.r = r
        self.q = q
        self.option_type = option_type
        

        self.tol = 0.001 # tolerance
        self.omega = 1.2 # relaxation factor

        self.S_max = 2.314 * self.K + 100 # max stock price

        self.N = 250 # number of time steps
        self.M = 100 # number of stock price steps
        self.dt = self.T / self.N # time step
        self.ds = self.S_max / self.M # stock price step

        self.I = np.arange(0, self.M + 1) # stock price array
        self.J = np.arange(0, self.N + 1) # time array

        self.old_val = np.zeros(self.M - 1) # old value
        self.new_val = np.zeros(self.M - 1) # new value

        self.payoff = self.get_payoff() # payoff
        self.old_layer = self.payoff # old layer
        self.bound_val = self.K * np.exp(-self.r * (self.N - self.J) * self.dt) # boundary value

        self.alpha = 0.25 * self.dt * (self.sigma ** 2 * (self.I ** 2) - (self.r - self.q) * self.I) # alpha
        self.alpha = self.alpha[1:] # remove first element
        self.beta = -self.dt * 0.5 * (self.sigma ** 2 * (self.I ** 2) + self.r) # beta
        self.beta = self.beta[1:] # remove first element
        self.gamma = 0.25 * self.dt * (self.sigma ** 2 * (self.I ** 2) + self.r * self.I) # gamma
        self.gamma = self.gamma[1:] # remove first element

        self.M2 = np.diag(1 + self.beta[:self.M - 1]) + np.diag(self.alpha[1:self.M - 1], k=-1) + np.diag(
            self.gamma[:self.M - 2], k=1) # matrix M2
        self.b = np.zeros(self.M - 1) # vector b

    def get_payoff(self, q=0):
        S = self.I[1:self.M] * self.ds
        if self.option_type == 'put': # payoff for put option
            payoff = np.maximum(self.K - S, 0)
        elif self.option_type == 'call':    # payoff for call option
            payoff = np.maximum(S - self.K, 0)
        else:
            raise ValueError("Invalid option type")
        
        # discounting for dividend yield
        if q != 0:
            t = (self.N - self.J) * self.dt
            d = np.exp(-q * t)
            payoff = payoff * d

        return payoff # return payoff

    def thomas_algorithm(self, a, b, c, d):
        # a: lower diagonal
        # b: main diagonal
        # c: upper diagonal
        # d: right hand side
        
        n = len(d)
        c_prime = np.zeros(n - 1)
        d_prime = np.zeros(n)
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]
        
        # forward sweep

        for i in range(1, n - 1):
            m = 1.0 / (b[i] - a[i - 1] * c_prime[i - 1])
            c_prime[i] = c[i] * m
            d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) * m
        
        # backward sweep
        
        x = np.zeros(n)
        x[-1] = d_prime[-1]
        
    
        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]

        return x # return x vector

    def get_price(self):
        # get price of option
        for j in range(self.N - 1, -1, -1):
            if self.option_type == 'call': # changed boundary condition for call option
                self.bound_val[j] = (self.S_max - self.K) * np.exp(
                    -self.r * (self.N - j) * self.dt)  # changed boundary condition for call option
            elif self.option_type == 'put': # existing boundary condition for put option
                self.bound_val[j] = self.K * np.exp(
                    -self.r * (self.N - j) * self.dt)  # existing boundary condition for put option

            self.b[0] = self.alpha[0] * (self.bound_val[j] + self.bound_val[j + 1])
            rhs = self.M2 @ self.old_layer + self.b # right hand side
            self.old_val = self.old_layer # old layer
            error = 1000000 
            while error > self.tol: # iterate until error is less than tolerance
                self.new_val[0] = self.old_val[0] + self.omega * (rhs[0] - (1 + self.beta[0]) * self.old_val[0] +
                                                                  self.gamma[0] * self.old_val[1]) / (

                                          1 + self.beta[0]) # first element of new value
                for k in range(1, self.M - 2): # iterate over all elements
                    self.new_val[k] = self.old_val[k] + (self.omega / (1 - self.beta[k])) * (
                            rhs[k] - (1 - self.beta[k]) * self.old_val[k] + self.alpha[k] * self.new_val[k - 1] +
                            self.gamma[k] * self.old_val[k + 1]) # new value
                self.new_val[self.M - 2] = self.old_val[self.M - 2] + (
                        self.omega / (1 - self.beta[self.M - 2])) * (
                                                   rhs[self.M - 2] - (1 - self.beta[self.M - 2]) *
                                                   self.old_val[self.M - 2] + self.alpha[self.M - 2] *
                                                   self.new_val[self.M - 3]) # last element of new value

                error = np.linalg.norm(self.new_val - self.old_val) # error
                self.old_val = self.new_val.copy()  # old value
            self.old_layer = self.new_val.copy() # old layer
        return self.old_layer[int(self.S / self.ds) - 1] # return price of option


# Define the parameters
T = 1
S = 100
K = 100
sigma = 0.6
r = 0.02
q = 0.01



