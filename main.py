
'''
Author: Jake Kemp

MATH 6204z

Project Description:

This project aims to provide a comprehensive analysis and comparison of various option pricing methods, 
for both European and American options. The main purpose of this project is to evaluate and understand 
the performance of these diverse methods in computing option prices, which could be useful in different 
financial and trading scenarios.

The following models and algorithms are used:

1. Closed-form Solutions: These are used for calculating European option prices.
2. Crank-Nicolson Finite Difference Method (FDM): Utilized for American options, with variations including Projected SOR and Regular SOR.
3. Crank-Nicolson Method with Successive Over-Relaxation (SOR) and Thomas Algorithm: Implemented for European options.
4. Explicit Finite Difference Method: Applied for both American and European options.
5. Implicit Finite Difference Method: Includes Brenn-Schwartz method and Projected SOR for American options, and SOR and Thomas algorithm for European options.
6. Monte Carlo Simulation: Deployed for calculating European option prices.
7. Regression Method II: Used for pricing American options.

The project further calculates errors between the computed prices and the closed-form solutions to evaluate the accuracy of these methods. The results are then tabulated and printed for further analysis.

Files Included:
1. main.py: The main driver script that imports all other modules and executes the computation.
2. closed_form_solutions.py: Contains functions for calculating European call and put option prices using closed-form solutions.
3. crank_nicholson_fdm_brennanschwartz.py: Holds the Crank-Nicolson method implementation with Brennan-Schwartz for American options.
4. crank_nicholson_fdm_projected_sor.py: Incorporates the Crank-Nicolson method with Projected SOR for American options.
5. crank_nicholson_fdm_sor_american.py: Contains the Crank-Nicolson method with regular SOR for American options.
6. crank_nicholson_fdm_sor_euro.py: Holds the Crank-Nicolson method with SOR for European options.
7. crank_nicholson_thomas_euro.py: Contains the Crank-Nicolson method with Thomas algorithm for European options.
8. explicit_american.py: Implements the Explicit FDM for American options.
9. explicit_euro.py: Implements the Explicit FDM for European options.
10. implicit_fdm_brenn_schwartz_american.py: Contains the Implicit FDM with Brenn-Schwartz for American options.
11. implicit_fdm_projected_sor_american.py: Contains the Implicit FDM with Projected SOR for American options.
12. implicit_fdm_sor.py: Implements the Implicit FDM with SOR for European options.
13. implicit_fdm_thomas.py: Implements the Implicit FDM with Thomas algorithm for European options.
14. monte_carlo_risk_neutral.py: Holds the Monte Carlo method for European options.
15. regression_method_II.py: Contains the Regression Method II for American options.


To use the code for different parameters, adjust the values of the parameters 
S, K, mu, r, delta, q, sigma, T, t0, N, M, num_paths, and num_steps in main.py. 
The code will then calculate the options prices using all the methods and display 
them in a tabular format along with the error (difference from the closed-form solution) 
for European options.

'''



import numpy as np
from tabulate import tabulate

from closed_form_solutions import closed_form_european_call, closed_form_european_put

from crank_nicholson_fdm_brennanschwartz import AmericanOption_crank_brennan
from crank_nicholson_fdm_projected_sor import AmericanOption_crank_projectedsor
from crank_nicholson_fdm_sor_american import AmericanOption_crank_regularsor
from crank_nicholson_fdm_sor_euro import EuropeanOption_crank_sor
from crank_nicholson_thomas_euro import EuropeanOption_thomas


from explicit_american import explicit_FDM_american
from explicit_euro import explicit_FDM_european

from implicit_fdm_brenn_schwartz_american import american_option_price_implicit_brenn_schwartz
from implicit_fdm_projected_sor_american import american_option_price_implicit_projected_sor
from implicit_fdm_sor import european_option_price_implicit_sor
from implicit_fdm_thomas import european_option_price_implicit_thomas

from monte_carlo_risk_neutral import calculate_european_option_price
from regression_method_II import regression_method_II_american_call, regression_method_II_american_put




S = 100
S_0 = S
K = 100
mu = 0.05
r = 0.02
delta = 0.
q = delta
sigma = 0.6
T = 1
t0 = 0
N = 100
M = 100
num_paths = 500  # Number of sample paths
num_steps = 800  # Number of steps (M)



# Calculate European option prices using closed-form solutions
european_call_price_closedform = closed_form_european_call(S, K, mu, r, delta, sigma, T, t0)
european_put_price_closedform = closed_form_european_put(S, K, mu, r, delta, sigma, T, t0)



# Calculate American option prices using Crank-Nicolson FDM with projected SOR
american_call_price_projectedsor = AmericanOption_crank_projectedsor(T, S, K, sigma, r, q, option_type='call').get_price()
american_put_price_projectedsor = AmericanOption_crank_projectedsor(T, S, K, sigma, r, q, option_type='put').get_price()



# Calculate American option prices using Crank-Nicolson FDM with regular SOR
american_call_price_regularsor = AmericanOption_crank_regularsor(T, S_0, K, sigma, r, q, option_type='call').get_price()
american_put_price_regularsor = AmericanOption_crank_regularsor(T, S_0, K, sigma, r, q, option_type='put').get_price()


# European Option using Crank-Nicolson method with Successive Over-Relaxation (SOR)
option_crank_sor = EuropeanOption_crank_sor(T, S, K, sigma, r, q, option_type='call')
call_price_crank_sor = option_crank_sor.get_price()

option_crank_sor = EuropeanOption_crank_sor(T, S, K, sigma, r, q, option_type='put')
put_price_crank_sor = option_crank_sor.get_price()


# European Option using Thomas algorithm
option_thomas = EuropeanOption_thomas(T, S, K, sigma, r, q, option_type='call')
call_price_thomas = option_thomas.get_price()

option_thomas = EuropeanOption_thomas(T, S, K, sigma, r, q, option_type='put')
put_price_thomas = option_thomas.get_price()


# American Option using Explicit Finite Difference Method
call_price_explicit_fdm_american = explicit_FDM_american(S, K, T, sigma, r, q, N, M, 'Call')
put_price_explicit_fdm_american = explicit_FDM_american(S, K, T, sigma, r, q, N, M, 'Put')


# European Option using Explicit Finite Difference Method
call_price_explicit_fdm_european = explicit_FDM_european(S, K, T, sigma, r, delta, N, M, 'Call')
put_price_explicit_fdm_european = explicit_FDM_european(S, K, T, sigma, r, delta, N, M, 'Put')


# American Option using Implicit Finite Difference Method (Brenn-Schwartz method)
call_price_implicit_brenn_schwartz = american_option_price_implicit_brenn_schwartz(K, T, sigma, r, q, S, 'call')
put_price_implicit_brenn_schwartz = american_option_price_implicit_brenn_schwartz(K, T, sigma, r, q, S, 'put')


# American Option using Implicit Finite Difference Method with Projected SOR
call_price_implicit_projected_sor = american_option_price_implicit_projected_sor(K, T, sigma, r, q, S, 'call')
put_price_implicit_projected_sor = american_option_price_implicit_projected_sor(K, T, sigma, r, q, S, 'put')



# Calculate the price of a European call option using the implicit finite difference method with SOR
european_call_option_implicit_sor = european_option_price_implicit_sor(K, T, sigma, r, q, S, 'call')


# Calculate the price of a European put option using the implicit finite difference method with SOR
european_put_option_implicit_sor = european_option_price_implicit_sor(K, T, sigma, r, q, S, 'put')


# Call the european_option_price_implicit_thomas function
european_call_option_thomas = european_option_price_implicit_thomas(K, T, sigma, r, q, S, 'call')
european_put_option_thomas = european_option_price_implicit_thomas(K, T, sigma, r, q, S, 'put')

# Calculate European option prices using Monte Carlo simulation
call_price_montecarlo = calculate_european_option_price(S, K, r, delta, sigma, T, 'call', num_paths, num_steps)
put_price_monte_carlo = calculate_european_option_price(S, K, r, delta, sigma, T, 'put', num_paths, num_steps)


# Calculate Calculate American option prices using regression method II
american_call_price_reg_II = regression_method_II_american_call(S, K, mu, r, delta, sigma, T, t0)
american_put_price_reg_II = regression_method_II_american_put(S, K, mu, r, delta, sigma, T, t0)



# Calculate the price of the American put option using the Crank-Nicolson method
american_put_price_crank_brennan = AmericanOption_crank_brennan(T, S, K, sigma, r, q).brennan_schwartz_algorithm()

# Calculate the price of the American call option using the Crank-Nicolson method
american_call_price_crank_brennan = AmericanOption_crank_brennan(T, S, K, sigma, r, q, option_type='call').brennan_schwartz_algorithm()





# Calculate errors
error_european_call_crank_sor = abs(european_call_price_closedform - call_price_crank_sor)
error_european_put_crank_sor = abs(european_put_price_closedform - put_price_crank_sor)

error_european_call_thomas = abs(european_call_price_closedform - call_price_thomas)
error_european_put_thomas = abs(european_put_price_closedform - put_price_thomas)

error_european_call_explicit_fdm = abs(european_call_price_closedform - call_price_explicit_fdm_european)
error_european_put_explicit_fdm = abs(european_put_price_closedform - put_price_explicit_fdm_european)

error_european_call_implicit_sor = abs(european_call_price_closedform - european_call_option_implicit_sor)
error_european_put_implicit_sor = abs(european_put_price_closedform - european_put_option_implicit_sor)

error_european_call_thomas_implicit = abs(european_call_price_closedform - european_call_option_thomas)
error_european_put_thomas_implicit = abs(european_put_price_closedform - european_put_option_thomas)

error_european_call_montecarlo = abs(european_call_price_closedform - call_price_montecarlo)
error_european_put_montecarlo = abs(european_put_price_closedform - put_price_monte_carlo)


# Prepare data for the European options table
european_table_data = [
    ["Closed-form", round(european_call_price_closedform, 6), "-", round(european_put_price_closedform, 6), "-"],
    ["Crank-Nicolson SOR", round(call_price_crank_sor, 6), round(error_european_call_crank_sor, 6), round(put_price_crank_sor, 6), round(error_european_put_crank_sor, 6)],
    ["Thomas Algorithm", round(call_price_thomas, 6), round(error_european_call_thomas, 6), round(put_price_thomas, 6), round(error_european_put_thomas, 6)],
    ["Explicit FDM", round(call_price_explicit_fdm_european, 6), round(error_european_call_explicit_fdm, 6), round(put_price_explicit_fdm_european, 6), round(error_european_put_explicit_fdm, 6)],
    ["Implicit FDM SOR", round(european_call_option_implicit_sor, 6), round(error_european_call_implicit_sor, 6), round(european_put_option_implicit_sor, 6), round(error_european_put_implicit_sor, 6)],
    ["Implicit FDM Thomas", round(european_call_option_thomas, 6), round(error_european_call_thomas_implicit, 6), round(european_put_option_thomas, 6), round(error_european_put_thomas_implicit, 6)],
    ["Monte Carlo", round(call_price_montecarlo, 6), round(error_european_call_montecarlo, 6), round(put_price_monte_carlo, 6), round(error_european_put_montecarlo, 6)],
]


# Prepare data for the American options table
american_table_data = [
    ["Crank-Nicolson FDM Projected SOR", round(american_call_price_projectedsor, 6), round(american_put_price_projectedsor, 6)],
    ["Crank-Nicolson FDM Regular SOR", round(american_call_price_regularsor, 6), round(american_put_price_regularsor, 6)],
    ["Explicit FDM", round(call_price_explicit_fdm_american, 6), round(put_price_explicit_fdm_american, 6)],
    ["Implicit FDM Brenn-Schwartz", round(call_price_implicit_brenn_schwartz, 6), round(put_price_implicit_brenn_schwartz, 6)],
    ["Implicit FDM Projected SOR", round(call_price_implicit_projected_sor, 6), round(put_price_implicit_projected_sor, 6)],
    ["Crank-Nicolson FDM Brennan-Schwartz", round(american_call_price_crank_brennan, 6), round(american_put_price_crank_brennan, 6)],
    ["Regression Method II", round(american_call_price_reg_II[0], 6), round(american_put_price_reg_II[0], 6)],
]

# Print tables
print("European Options")
print(tabulate(european_table_data, headers=["Method", "Call Price", "Call Error", "Put Price", "Put Error"], tablefmt="grid"))

print("\nAmerican Options")
print(tabulate(american_table_data, headers=["Method", "Call Price", "Put Price"], tablefmt="grid"))
