import numpy as np
from scipy.optimize import differential_evolution, minimize
from ppg_basis.utils.generator_utils import *
from ppg_basis.model import *
from ppg_basis.cost import objective_function
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ppgExtractor:
    def __init__(self, signal: np.ndarray, fs: float, hr: float, sigma: float, L: int,
                 basis_type: str, mse_flag: bool = True, corr_flag: bool = True, appg_flag: bool = False):
        """
        Constructor for Extractor Class
        :param signal: Input signal to analyze
        :param fs: sampling rate (Hz)
        :param hr: Heart Rate (BPM)
        :param sigma: Standard Deviation in HR
        :param L: Number of Basis Functions
        :param basis_type: Basis function (gaussian, gamma, or skewed-gaussian)
        :param mse_flag: Cost includes mean-squared-error
        :param corr_flag: Cost includes (1-corr)
        :param appg_flag: Cost includes normalized root-mean-square error of second derivative of PPG
        """
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

    def extract_ppg(self, block_update: bool = True, coord_cycles: int = 4):
        """
        Extract PPG params (theta and basis specific)
        :param block_update: Flag for whether to run block update per basis in phase 2 optimization
        :param coord_cycles: If block update flag is True, dictates how many cycles per basis
        :return: thetas and params from phase 2 optimization
        """
        # flatten the initial guess
        P = self.params.shape[1]
        x0 = np.concatenate([self.thetai, self.params.flatten()])
        # Differential Evolution
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
        x_phase1 = sls_res.x
        theta_phase1 = x_phase1[:self.L]
        params_phase1 = x_phase1[self.L:].reshape((self.L, P))

        theta_phase2 = theta_phase1.copy()
        params_phase2 = params_phase1.copy()
        if block_update:
            for _ in range(coord_cycles):
                for i in range(self.L):
                    xi0 = np.concatenate([[theta_phase2[i]], params_phase2[i]])

                    # Define cost for the i-th block, keeping others fixed
                    def block_cost(xi):
                        thetas = theta_phase2.copy()
                        params = params_phase2.copy()
                        thetas[i] = xi[0]
                        params[i] = xi[1:]
                        x_full = np.concatenate([thetas, params.flatten()])
                        return self.get_cost(x_full)

                    # Define bounds for i-th block
                    block_bounds = [self.bounds[i]] + self.bounds[self.L + i * P: self.L + (i + 1) * P]

                    # Local optimize for block
                    res = minimize(
                        fun=block_cost,
                        x0=xi0,
                        bounds=block_bounds,
                        method='SLSQP',
                        options={'maxiter': 300, 'ftol': 1e-6}
                    )

                    # Update parameters
                    theta_phase2[i] = res.x[0]
                    params_phase2[i] = res.x[1:]
        else:
            res = minimize(fun=self.get_cost,
                           x0=x_phase1,
                           bounds=self.bounds,
                           constraints=self.constraints,
                           method='SLSQP',
                           options={'maxiter': 1000, 'ftol': 1e-8})
            theta_phase2 = res.x[:self.L]
            params_phase2 = res.x[self.L:].reshape((self.L, P))
        return theta_phase2, params_phase2

    def plot_cost_landscape(self, mse_flag: bool = True, corr_flag: bool = False, appg_flag: bool = False, resolution: int = 10):
        P = self.params.shape[1]
        theta_bounds = self.bounds[:self.L]
        param_bounds = self.bounds[self.L:]

        for i in range(self.L):
            print(f"Basis {i + 1}/{self.L}")

            # Parameter ranges
            theta_range = np.linspace(theta_bounds[i][0], theta_bounds[i][1], resolution)
            p_ranges = [
                np.linspace(param_bounds[i * P + j][0], param_bounds[i * P + j][1], resolution)
                for j in range(P)
            ]
            fixed_vals = [np.mean(param_bounds[i * P + j]) for j in range(P)]

            X, Y, Z, C, S = [], [], [], [], []

            for theta in theta_range:
                for p1 in p_ranges[0]:
                    for p2 in p_ranges[1]:
                        for p3 in (p_ranges[2] if P > 2 else [None]):
                            for p4 in (p_ranges[3] if P > 3 else [None]):
                                thetas = self.thetai.copy()
                                thetas[i] = theta
                                params = self.params.copy()

                                param_vec = [p1, p2]
                                if P > 2:
                                    param_vec.append(p3 if p3 is not None else fixed_vals[2])
                                if P > 3:
                                    param_vec.append(p4 if p4 is not None else fixed_vals[3])
                                params[i] = param_vec

                                x_full = np.concatenate([thetas, params.flatten()])
                                model_ppg = unified_model_ode(
                                    ppinterval=self.pp_interval,
                                    fs=self.fs,
                                    seconds=self.rr_interval,
                                    basis_type=self.basis_type,
                                    thetai=thetas,
                                    basis_params=params
                                )

                                cost_total = objective_function(model_ppg, self.signal,
                                                                mse_flag=mse_flag,
                                                                corr_flag=corr_flag,
                                                                appg_flag=appg_flag)

                                X.append(theta)
                                Y.append(p1)
                                Z.append(p2)
                                # Size reflects 4th parameter if present
                                if P > 3:
                                    size_val = p4
                                elif P > 2:
                                    size_val = p3
                                else:
                                    size_val = 0.5  # default size
                                C.append(cost_total)
                                S.append(40 + 80 * size_val)

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(X, Y, Z, c=C, s=S, cmap='plasma', alpha=0.8)
            ax.set_title(f"Full Cost Landscape for Basis {i + 1}")
            ax.set_xlabel("Theta")
            ax.set_ylabel("Param 1")
            ax.set_zlabel("Param 2")
            fig.colorbar(sc, ax=ax, shrink=0.6, label="Total Cost")
            plt.tight_layout()
            plt.show()