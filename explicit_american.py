import numpy as np

def explicit_FDM_american(S, K, T, sigma, r, q, N, M, CallPut):
    '''
    Explicit finite difference method for pricing American call and put options

    Parameters:
        S - intial price of underlying asset
        K - strike price
        T - time to maturity
        sigma - volatility
        r - risk-free rate
        delta - dividend rate
        N - number of spacing points along the time partition (horizontal)
        M - number of partition points from 0 to the upper/lower boundary (vertical)
        CallPut - 'Call' or 'Put'

    The "explicit_FDM_american" function is an implementation of the explicit finite difference method used to price American
    call and put options. It takes in various parameters including the initial price of the underlying asset (S), strike price
    (K), time to maturity (T), volatility (sigma), risk-free rate (r), dividend rate (delta), number of spacing points along
    the time partition (N), number of partition points from 0 to the upper/lower boundary (M), and the type of option (CallPut).
    The function returns the estimated option price computed by the finite difference grid, taking into account early exercise.
    To ensure convergence of the method, the condition dx >= sigmasqrt(3dt) must be satisfied. The function uses the best choice
    for dx, which is dx = sigmasqrt(3dt), and solves for the number of time intervals (N) by setting (sigma^2)3dt + dt = epsilon
    or dt = epsilon/(1 + 3*sigma^2) for a given error epsilon.

    '''
    dt = T / N
    dx = sigma * np.sqrt(3 * dt)
    nu = r - q - 0.5 * sigma ** 2
    pu = 0.5 * dt * ((sigma / dx) ** 2 + nu / dx)
    pm = 1.0 - dt * (sigma / dx) ** 2 - r * dt 
    pd = 0.5 * dt * ((sigma / dx) ** 2 - nu / dx)
    grid = np.zeros((N + 1, 2 * M + 1))

    # Asset prices at maturity:
    St = [S * np.exp(-M * dx)]
    for j in range(1, 2 * M + 1):
        St.append(St[j - 1] * np.exp(dx))

    # Option value at maturity:
    for j in range(2 * M + 1):
        if CallPut == 'Call':
            grid[N, j] = max(0, St[j] - K)
        elif CallPut == 'Put':
            grid[N, j] = max(0, K - St[j])

    # Backwards computing through grid:
    for i in range(N - 1, -1, -1):
        for j in range(1, 2 * M):
            # Compute option value without early exercise
            value_without_exercise = pu * grid[i + 1, j + 1] + pm * grid[i + 1, j] + pd * grid[i + 1, j - 1]

            # Compute option value with early exercise
            if CallPut == 'Call':
                value_with_exercise = St[j] - K
            elif CallPut == 'Put':
                value_with_exercise = K - St[j]
                
            # Take the maximum of the two values
            grid[i, j] = max(value_without_exercise, value_with_exercise)
            
    # Return the option price at the initial time
    return grid[0, M]

# Test the function
S = 100
K = 100
T = 1
sigma = 0.6
r = 0.02
q = 0.01
N = 100
M = 100



