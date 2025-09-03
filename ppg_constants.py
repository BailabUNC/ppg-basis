# Constant values used throughout the codebase

default_M = 512

# used to generate bounds and constraints for optimization
param_bnds_dict = {
    'gaussian' : [(0.0, 1.0), (0.05, 3.0)],
    'gamma' : [(0.0, 1.0), (1.0, 6.0), (0.05, 3.0)],
    'skewed-gaussian' : [(0.0, 1.0), (0.05, 3.0), (-10.0, 10.0)]
}

basis_types = ['gaussian', 'gamma', 'skewed-gaussian']

default_solver = "basis"

# default parameters
default_params = {
    # shared parameters
    "fs" : 60,
    "basis_type" : "gaussian",
    "L" : 2,
    "mu" : 1,
    "sigma" : 0,
    "solver" : default_solver,

    # extractor-specific parameters
    "cost_metrics" : ["mse", "corr"],
    "cost_func" : None,

    # generator-specific parameters
    "hr" : 60,
    "duration" : 1,
}

param_validators = {
    # shared parameters
    "fs" : lambda x: isinstance(x, int) and x > 0,
    "basis_type" : lambda x: x in basis_types,
    "L" : lambda x : isinstance(x, int) and x > 1,
    "mu" : lambda x: x > 0,
    "sigma" : lambda x : x > 0,
    "solver" : lambda x : isinstance(x, str),

    # extractor-specific parameters
    "cost_metrics" : lambda x : isinstance(x, list) and all(isinstance(metric, str) for metric in x),
    "cost_func" : lambda x : isinstance(x, callable),

    # generator-specific parameters
    "hr" : lambda x : isinstance(x, int) and x > 0,
    "duration" : lambda x : x > 0,
}