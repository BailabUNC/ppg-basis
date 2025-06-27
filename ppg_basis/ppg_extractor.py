import numpy as np
from scipy.optimize import differential_evolution, minimize
from ppg_basis.utils.generator_utils import pp_interval_generator
from ppg_basis.model import *
from ppg_basis.cost import *

class ppgExtractor():
    def __init__(self, signal, fs, hr, sigma, L, basis_type):
        self.signal = signal
        self.fs = fs
        self.basis_type=basis_type
        self.rr_interval = len(signal)/fs
        self.pp_interval = pp_interval_generator(time=self.rr_interval,
                                                 mu=60/hr,
                                                 sigma=sigma)
        self.thetai, self.params = generate_basis_parameters(L=L,
                                                             basis_type=basis_type,
                                                             random_state=None)
        self.bounds, self.constraints = get_bounds_and_constraints(L=L,
                                                                   basis_type=basis_type)
        # TODO: Input checker should happen here

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
