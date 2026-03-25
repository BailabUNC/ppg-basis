from .model_basis import unified_model_basis
from .model_template import unified_model_template
from .model_fft import unified_model_fft
from .model_ode import unified_model_ode
from ppg_basis.ppg_constants import default_solver, default_M

try:
    import fastppg as _fastppg
    _HAS_FASTPPG = True
except ImportError:
    _fastppg = None
    _HAS_FASTPPG = False

import numpy as np
import warnings


def unified_model(ppinterval, fs, seconds, basis_type, thetai, basis_params, solver=default_solver, M=default_M):
    """
    Unified entry point:
      - mode="basis"  → closed-form basis synthesis (no ODE)
      - mode="ode"    → RK ODE (ode_solver in {"rk3","rk4"})
    """
    # Attempt fast path
    if _HAS_FASTPPG:
        try:
            return _fastppg.forward_model(
                pp=np.asarray(ppinterval, dtype=np.float64),
                fs=int(fs), seconds=float(seconds),
                basis_type=basis_type,
                thetai=np.asarray(thetai, dtype=np.float64),
                params=np.asarray(basis_params, dtype=np.float64),
                solver=solver, M=int(M)
            )
        except Exception as e:
            warnings.warn(
                f"fastppg.forward_model failed ({e}); falling back to Python solver.",
                RuntimeWarning
            )

    if solver == "basis":
        return unified_model_basis(ppinterval, fs, seconds, basis_type, thetai, basis_params, M)
    elif solver == 'template':
        return unified_model_template(ppinterval, fs, seconds, basis_type, thetai, basis_params, M)
    elif solver == "fft":
        return unified_model_fft(ppinterval, fs, seconds, basis_type, thetai, basis_params, M)
    elif solver == "rk3" or solver == "rk4":
        return unified_model_ode(ppinterval, fs, seconds, basis_type, thetai, basis_params, solver, M)
    else:
        raise ValueError(f"Unsupported solver: {solver}")