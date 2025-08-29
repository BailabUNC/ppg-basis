# Constant values used throughout the codebase

default_M = 1024

# used to generate bounds and constraints for optimization
param_bnds_dict = {
    'gaussian' : [(0.0, 1.0), (0.05, 3.0)],
    'gamma' : [(0.0, 1.0), (1.0, 6.0), (0.05, 3.0)],
    'skewed-gaussian' : [(0.0, 1.0), (0.05, 3.0), (-10.0, 10.0)]
}

basis_types = ['gaussian', 'gamma', 'skewed-gaussian']

# default parameters
default_params = {

    # shared parameters
    "fs" : 60,
    "basis_type" : "gaussian",
    "L" : 2,
    "mu" : 1,
    "sigma" : 0,
    "solver" : "rk4",

    # extractor-specific parameters
    "cost_metrics" : ["mse", "corr"],

    # generator-specific parameters
    "hr" : 60,
    "duration" : 1,
}