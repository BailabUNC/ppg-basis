import numpy as np
from scipy.optimize import differential_evolution, minimize
from ppg_basis.utils.generator_utils import *
from ppg_basis.model import *
from ppg_basis.cost import objective_function

class ppgExtractor:
    def __init__(self, signal: np.ndarray, fs: float, hr: float, sigma: float, L: int,
                 basis_type: str, mse_flag: bool = True, corr_flag: bool = True, appg_flag: bool = False):
        self.signal = signal
        self.fs = fs
        self.basis_type = basis_type
        self.L = L

        # cost‐function flags
        self.mse_flag = mse_flag
        self.corr_flag = corr_flag
        self.appg_flag = appg_flag

        # build RR‐interval & initial basis
        self.rr_interval = len(signal) / fs
        self.pp_interval = pp_interval_generator(time=self.rr_interval,
                                                 mu=60/hr,
                                                 sigma=sigma)

        # random initial thetas & params
        self.thetai, self.params = generate_basis_parameters(L=L,
                                                             basis_type=basis_type,
                                                             random_state=None)

        # bounds & constraints for flat vector of length = L*P
        self.bounds, self.constraints = get_bounds_and_constraints(L=L,
                                                                   basis_type=basis_type)

    def get_cost(self, x: np.ndarray) -> float:
        """
        x is a flat array of length = L * (#params per basis).
        We reshape it, run the forward model, and compute the scalar cost.
        """
        # reshape into (L, P)
        P = self.params.shape[1]
        theta_new = x[:self.L]
        params_new = x[self.L:].reshape((self.L, P))

        # simulate
        model_ppg = unified_model_ode(ppinterval=self.pp_interval,
                                     fs=self.fs,
                                     seconds=self.rr_interval,
                                     basis_type=self.basis_type,
                                     thetai=theta_new,
                                     basis_params=params_new)

        # scalar cost
        return objective_function(model=model_ppg,
                                 signal=self.signal,
                                 mse_flag=self.mse_flag,
                                 corr_flag=self.corr_flag,
                                 appg_flag=self.appg_flag)

    def extract_ppg(self):
        # flatten the initial guess
        P = self.params.shape[1]
        x0 = np.concatenate([self.thetai, self.params.flatten()])
        # Differential Evolutionx0 = self.params.flatten()
        de_res = differential_evolution(func=self.get_cost,
                                        bounds=self.bounds,
                                        maxiter=60,
                                        popsize=12,
                                        tol=1e-2,
                                        polish=False)

        # local refinement (SLSQP)
        sls_res = minimize(fun=self.get_cost,
                           x0=de_res.x,
                           bounds=self.bounds,
                           constraints=self.constraints,
                           method='SLSQP',
                           options={'maxiter': 1000, 'ftol': 1e-8})

        # reshape optimized params back to (L, P)
        x_opt = sls_res.x
        theta_opt = x_opt[:self.L]
        params_opt = x_opt[self.L:].reshape((self.L, P))
        return theta_opt, params_opt