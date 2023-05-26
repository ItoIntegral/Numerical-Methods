# Numerical-Methods
Numerical methods for option pricing

# Description
This project provides a comprehensive analysis and comparison of various option pricing methods for both European and American options. The aim is to evaluate and understand the performance of these methods in computing option prices, useful in different financial and trading scenarios.

The implemented models and algorithms are as follows:

- Closed-form Solutions: Used for calculating European option prices.

- Crank-Nicolson Finite Difference Method (FDM): Applied to American options, with variations including Projected SOR and Regular SOR.

- Crank-Nicolson Method with Successive Over-Relaxation (SOR) and Thomas Algorithm: Used for European options.

- Explicit Finite Difference Method: Used for both American and European options.

- Implicit Finite Difference Method: Includes Brenn-Schwartz method and Projected SOR for American options, and SOR and Thomas algorithm for European options.

- Monte Carlo Simulation: Used for calculating European option prices.

- Regression Method II: Used for pricing American options.

The project computes option prices using these methods and compares them to closed-form solutions, further calculating the errors between computed prices and the closed-form solutions to evaluate the accuracy of these methods. The results are presented in a tabular format for further analysis.

# Usage
To use the code for different parameters, adjust the values of the parameters S, K, mu, r, delta, q, sigma, T, t0, N, M, num_paths, and num_steps in main.py. The code will then calculate the options prices using all the methods and display them in a tabular format along with the error (difference from the closed-form solution) for European options.

# Files Structure
The project structure is as follows:

- main.py: Main driver script that imports all other modules and executes the computation.

- closed_form_solutions.py: Functions for calculating European call and put option prices using closed-form solutions.

- crank_nicholson_fdm_brennanschwartz.py: Crank-Nicolson method implementation with Brennan-Schwartz for American options.

- crank_nicholson_fdm_projected_sor.py: Crank-Nicolson method with Projected SOR for American options.

- crank_nicholson_fdm_sor_american.py: Crank-Nicolson method with regular SOR for American options.

- crank_nicholson_fdm_sor_euro.py: Crank-Nicolson method with SOR for European options.

- crank_nicholson_thomas_euro.py: Crank-Nicolson method with Thomas algorithm for European options.

- explicit_american.py: Explicit FDM for American options.

- explicit_euro.py: Explicit FDM for European options.

- implicit_fdm_brenn_schwartz_american.py: Implicit FDM with Brenn-Schwartz for American options.

- implicit_fdm_projected_sor_american.py: Implicit FDM with Projected SOR for American options.

- implicit_fdm_sor.py: Implicit FDM with SOR for European options.

- implicit_fdm_thomas.py: Implicit FDM with Thomas algorithm for European options.

- monte_carlo_risk_neutral.py: Monte Carlo method for European options.

- regression_method_II.py: Regression Method II for American options.


For a full working example, check the main.py file.

