from ppg_basis.model import unified_solver
from ppg_basis.utils.ppg_utils import pp_interval_generator, generate_basis_parameters, validate_param
from ppg_constants import default_M

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
        self.fs = validate_param("fs", fs)
        self.hr = validate_param("hr", hr)
        self.mu = validate_param("mu", mu)
        self.sigma = validate_param("sigma", sigma)
        self.duration = validate_param("duration", duration)
        self.L = validate_param("L", L)
        self.basis_type = validate_param("basis_type", basis_type)
        self.solver = validate_param("solver", solver)

        if thetas is not None and params is not None:
            self.thetai, self.params = thetas, params
        else:
            self.thetai, self.params = generate_basis_parameters(L = self.L, basis_type = self.basis_type)

        self.ppinterval = pp_interval_generator(time=self.duration,
                                                mu=self.mu,
                                                sigma=self.sigma)
        self.signal = None

    def generate_signal(self, M):
        """
        Generates PPG signal
        :return: z(t)
        """
        if not isinstance(M, int):
            M = default_M
        self.signal = unified_solver.unified_model(ppinterval=self.ppinterval,
                                                   fs=self.fs,
                                                   seconds=self.duration,
                                                   basis_type=self.basis_type,
                                                   thetai=self.thetai,
                                                   basis_params=self.params,
                                                   solver=self.solver,
                                                   M=M)
        return self.signal