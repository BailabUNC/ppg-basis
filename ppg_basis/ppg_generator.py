from ppg_basis.model import unified_solver
from ppg_basis.utils.ppg_utils import pp_interval_generator, generate_basis_parameters
from ppg_constants import default_params, basis_types

class ppgGenerator():
    def __init__(self,
                 fs,
                 hr,
                 mu,
                 sigma,
                 duration,
                 L,
                 basis_type,
                 solver: str = "rk3",
                 thetas=None,
                 params=None):
        """
        Constructor for Generator Class
        :param fs: sampling rate (Hz)
        :param hr: Heart Rate (BPM)
        :param mu: mean pulse-to-pulse interval variation
        :param sigma: Standard Deviation in HR
        :param duration: total time of PPG window
        :param L: Number of Basis Functions
        :param basis_type: Basis function (gaussian, gamma, or skewed-gaussian)
        :param solver: method of ODE solving (generally an n-th order RK method)
        :param thetas: phase location in PPG period
        :param params: basis parameter list
        """
        self.fs = fs if fs > 0 else default_params["fs"]
        self.hr = hr if hr > 0 else default_params["hr"] # FIXME: hr is never accessed
        self.mu = mu if mu > 0 else default_params["mu"]
        self.sigma = sigma if sigma > 0 else default_params["sigma"]
        self.duration = duration if duration > 0 else default_params["duration"]
        self.L = L if L > 1 else default_params["L"]
        self.basis_type = basis_type if basis_type in basis_types else default_params["basis_type"]
        self.solver = solver if isinstance(solver, str) else default_params["solver"]
        
        self.thetai, self.params = thetas, params if thetas and params is not None else generate_basis_parameters(L = self.L, basis_type = self.basis_type)

        self.ppinterval = pp_interval_generator(time=self.duration,
                                                mu=self.mu,
                                                sigma=self.sigma)
        self.signal = None

    def generate_signal(self):
        """
        Generates PPG signal
        :return: z(t)
        """
        self.signal = unified_solver(ppinterval=self.ppinterval,
                                        fs=self.fs,
                                        seconds=self.duration,
                                        basis_type=self.basis_type,
                                        thetai=self.thetai,
                                        basis_params=self.params,
                                        solver=self.solver)
        return self.signal