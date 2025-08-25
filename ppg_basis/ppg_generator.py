from ppg_basis import model
from ppg_basis.utils.ppg_utils import pp_interval_generator

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
        self.fs =fs
        self.hr = hr
        self.mu = mu
        self.sigma = sigma
        self.duration = duration
        self.L = L
        self.basis_type = basis_type
        self.solver = solver

        if thetas is None or params is None:
            self.thetai, self.params = model.generate_basis_parameters(L=self.L,
                                                                       basis_type=self.basis_type)
        else:
            self.thetai = thetas
            self.params = params

        self.ppinterval = pp_interval_generator(time=self.duration,
                                                mu=self.mu,
                                                sigma=self.sigma)
        self.signal = None

    def generate_signal(self):
        """
        Generates PPG signal
        :return: z(t)
        """
        self.signal = model.unified_model(ppinterval=self.ppinterval,
                                          fs=self.fs,
                                          seconds=self.duration,
                                          basis_type=self.basis_type,
                                          thetai=self.thetai,
                                          basis_params=self.params,
                                          solver=self.solver)
        return self.signal