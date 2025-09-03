import numpy as np
from scipy.optimize import differential_evolution, minimize
from ppg_basis.model.unified_solver import unified_model
from ppg_basis.utils.ppg_utils import *
from ppg_basis.cost import objective_function
import fastplotlib as fpl
from ipywidgets import IntSlider, Checkbox, VBox, HTML
from ppg_constants import default_params

class ppgExtractor:
    def __init__(self,
                 signal: np.ndarray,
                 fs: float,
                 hr: float,
                 sigma: float,
                 L: int,
                 basis_type: str,
                 solver: str,
                 cost_metrics: list,
                 cost_func: callable):
        """
        Constructor for Extractor Class
        :param signal: Input signal to analyze
        :param fs: sampling rate (Hz)
        :param hr: Heart Rate (BPM)
        :param sigma: Standard Deviation in HR
        :param L: Number of Basis Functions
        :param basis_type: Basis function (gaussian, gamma, or skewed-gaussian)
        :param solver: method of ODE solving (generally an n-th order RK method)
        :param cost_metrics: cost metrics to be added to objective func
        :param cost_func: cost function to be added to objective func
        """
        if signal is None or len(signal) > 3:
            raise ValueError("signal dim cannot exceed 3")
        self.signal = signal

        self.fs = validate_param("fs", fs)
        self.basis_type = validate_param("basis_type", basis_type)
        self.L = validate_param("L", L)
        self.solver = validate_param("solver", solver)

        # cost‐function flags
        self.cost_metrics = validate_param("cost_metrics", cost_metrics)
        self.cost_func = validate_param("cost_func", cost_func)

        # build RR‐interval & initial basis
        self.rr_interval = len(signal) / self.fs
        self.pp_interval = pp_interval_generator(time = self.rr_interval,
                                                 mu = 60 / validate_param("hr", hr),
                                                 sigma = validate_param("sigma", sigma))
        
        # random initial thetas & params
        self.thetai, self.params = generate_basis_parameters(L = self.L,
                                                             basis_type = self.basis_type,
                                                             random_state = None)
        
        # bounds & constraints for flat vector of length = L*P
        self.bounds, self.constraints = get_bounds_and_constraints(L = self.L,
                                                                   basis_type = self.basis_type)

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
        model_ppg = unified_model(ppinterval=self.pp_interval,
                                      fs=self.fs,
                                      seconds=self.rr_interval,
                                      basis_type=self.basis_type,
                                      thetai=theta_new,
                                      basis_params=params_new,
                                      solver=self.solver)

        # scalar cost
        return objective_function(model=model_ppg,
                                 signal=self.signal,
                                 cost_metrics=self.cost_metrics,
                                 func=self.cost_func)

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
                    res = minimize(fun=block_cost,
                                   x0=xi0,
                                   bounds=block_bounds,
                                   method='SLSQP',
                                   options={'maxiter': 300, 'ftol': 1e-6})

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

    def _generate_cost_landscape(self,
                                 basis_index: int,
                                 resolution: int):
        """
        Generate full cost grid for a single basis.
        Returns X, Y, Z, C, S arrays.
        """
        P = self.params.shape[1]
        # bounds for theta_i and its P parameters
        theta_min, theta_max = self.bounds[basis_index]
        p_bounds = self.bounds[
            self.L + basis_index*P : self.L + (basis_index+1)*P
        ]

        theta_vals = np.linspace(theta_min, theta_max, resolution)
        p_ranges = [np.linspace(b[0], b[1], resolution) for b in p_bounds]
        defaults = [np.mean(b) for b in p_bounds]

        thetai = self.thetai.copy()
        params = self.params.copy()

        X, Y, Z, C, S = [], [], [], [], []
        for θ in theta_vals:
            for p1 in p_ranges[0]:
                for p2 in p_ranges[1]:
                    for p3 in (p_ranges[2] if P>2 else [None]):
                        for p4 in (p_ranges[3] if P>3 else [None]):
                            thetai[basis_index] = θ
                            vec = [p1, p2]
                            if P>2:
                                vec.append(p3 if p3 is not None else defaults[2])
                            if P>3:
                                vec.append(p4 if p4 is not None else defaults[3])
                            params[basis_index] = vec

                            model_ppg = unified_model(ppinterval=self.pp_interval,
                                                      fs=self.fs,
                                                      seconds=self.rr_interval,
                                                      basis_type=self.basis_type,
                                                      thetai=thetai,
                                                      basis_params=params,
                                                      solver=self.solver)
                            cost_val = objective_function(model_ppg, self.signal,
                                                          cost_metrics = self.cost_metrics)

                            X.append(θ)
                            Y.append(p1)
                            Z.append(p2)
                            C.append(cost_val)
                            # record extra parameter for slicing
                            if P>3:
                                S.append(p4)
                            elif P>2:
                                S.append(p3)
                            else:
                                S.append(0.0)
        return (
            np.array(X), np.array(Y), np.array(Z),
            np.array(C), np.array(S)
        )

    def plot_cost_landscape(self, resolution: int = 10) -> list:
        """
        Displays an interactive Fastplotlib viewer for each basis.
        Returns a list of ipywidget.VBox containers.
        """
        containers = []
        for i in range(self.L):
            print(f"Basis {i+1}/{self.L}")
            X, Y, Z, C_raw, S = self._generate_cost_landscape(
                basis_index=i,
                resolution=resolution
            )
            # normalize cost to [0,1]
            C = (C_raw - C_raw.min()) / (C_raw.max() - C_raw.min())

            coords = np.column_stack([X, Y, Z])
            unique_S = sorted(set(S))
            slice_masks = [(S==sv) for sv in unique_S]

            # build figure
            fig = fpl.Figure(shape=(1,1),
                             cameras="3d",
                             controller_types="orbit",
                             size=(700,560))
            mask0 = slice_masks[0]
            scatter_ref = [
                fig[0,0].add_scatter(
                    coords[mask0],
                    cmap='viridis',
                    cmap_transform=C[mask0],
                    sizes=10
                )
            ]
            checkbox_auto = Checkbox(False, description="autoscale")
            slider_idx = IntSlider(
                value=0, min=0,
                max=len(unique_S)-1,
                step=1,
                description="slice idx",
                continuous_update=True
            )
            status = HTML()

            def _update(change,
                        fig=fig,
                        coords=coords,
                        C=C,
                        masks=slice_masks,
                        scatter_ref=scatter_ref,
                        checkbox=checkbox_auto,
                        status=status,
                        basis=i,
                        uniq_S=unique_S):
                idx = change['new']
                mask = masks[idx]
                # remove + re‑add
                fig[0,0].remove_graphic(scatter_ref[0])
                scatter_ref[0] = fig[0,0].add_scatter(
                    coords[mask],
                    cmap='viridis',
                    cmap_transform=C[mask],
                    sizes=10
                )
                if checkbox.value:
                    fig[0,0].auto_scale(maintain_aspect=False)
                status.value = (
                    f"Basis {basis+1}, slice {idx}, S={uniq_S[idx]:.3f}"
                )

            slider_idx.observe(_update, names=['value'])
            # initial draw
            _update({'new': 0})

            containers.append(
                VBox([fig.show(), slider_idx, checkbox_auto, status])
            )
        return containers
