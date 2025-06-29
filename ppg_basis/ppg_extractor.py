import warnings
import numpy as np
from scipy.optimize import differential_evolution, minimize
from ppg_basis.utils.generator_utils import pp_interval_generator
from ppg_basis.model import *
from ppg_basis.cost import *
from typing import Callable

class ppgExtractor():
    def __init__(self, signal, fs, hr, sigma, L, basis_type):
        self.signal, self.fs, mu, validated_sigma, validated_L, self.basis_type = ppgExtractor._validateParams(
            signal, 
            fs, 
            hr, 
            sigma, 
            L, 
            basis_type)
        self.rr_interval = len(signal)/fs
        self.pp_interval = pp_interval_generator(time=self.rr_interval,
                                                 mu=mu,
                                                 sigma=validated_sigma)
        self.thetai, self.params = generate_basis_parameters(L=validated_L,
                                                             basis_type=basis_type,
                                                             random_state=None)
        self.bounds, self.constraints = get_bounds_and_constraints(L=validated_L,
                                                                   basis_type=basis_type)
        
    @staticmethod
    def _validateParams(signal, fs, hr, sigma, L, basis_type):
        return [
            ppgExtractor._validate(signal, lambda x : x != None and len(x) != 3, "signal"), # NOTE: No default value here throws ValueError in _validate
            ppgExtractor._validate(fs, lambda x: x > 0, "fs", 60),
            60 / ppgExtractor._validate(hr, lambda x: x > 0, "hr", 60),
            ppgExtractor._validate(sigma, lambda x: x > 0, "sigma", 0),
            ppgExtractor._validate(L, lambda x: x > 1, "L", 2), # NOTE: set default number of basis functions to 2
            ppgExtractor._validate(basis_type, lambda x: x in ['gaussian', 'gamma', 'skewed-gaussian'], 'basis_type', 'gaussian') # NOTE: set default basis type to gaussian
        ]
    
    @staticmethod
    def _validate(value: any, constraint: Callable, name: str, default: any = None):
        if constraint(value):
            return value
        if default is None:
            raise ValueError("Invalid argument for value {name}: {value} and no default available")
        warnings.warn(f'{name} parameter is invalid ({value}), defaulting to {default}', UserWarning)
        return default

    def get_cost(self, mse_flag: bool=True, corr_flag: bool=True, appg_flag: bool=False):
        model = unified_model_ode(ppinterval=self.pp_interval,
                                  fs=self.fs,
                                  seconds=self.rr_interval,
                                  basis_type=self.basis_type,
                                  thetai=self.thetai,
                                  basis_params=self.params)
        cost = objective_function(model=model,
                                  signal=self.signal,
                                  mse_flag=mse_flag,
                                  corr_flag=corr_flag,
                                  appg_flag=appg_flag)
        return cost

    def extract_ppg(self):
        diff_ev_results = differential_evolution(self.get_cost,
                                                 bounds=self.bounds,
                                                 maxiter=60,
                                                 popsize=12,
                                                 tol=1e-2,
                                                 polish=False)
        sls_results = minimize(self.get_cost,
                               diff_ev_results.x,
                               bounds=self.bounds,
                               constraints=self.constraints,
                               method='SLSQP',
                               options={'maxiter':1000, 'ftol':1e-8})
        params_results = sls_results.x.copy()
        return params_results