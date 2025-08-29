from model_basis import unified_model_basis
from model_template import unified_model_template
from model_fft import unified_model_fft
from model_ode import unified_model_ode
from ppg_constants import default_M, default_solver

def unified_model(ppinterval, fs, seconds, basis_type, thetai, basis_params, solver = default_solver):
    """
    Unified entry point:
      - mode="basis"  → closed-form basis synthesis (no ODE)
      - mode="ode"    → RK ODE (ode_solver in {"rk3","rk4"})
    """
    if solver == "basis":
        return unified_model_basis(ppinterval, fs, seconds, basis_type, thetai, basis_params)
    elif solver == 'template':
        return unified_model_template(ppinterval, fs, seconds, basis_type, thetai, basis_params, M = default_M)
    elif solver == "fft":
        return unified_model_fft(ppinterval, fs, seconds, basis_type, thetai, basis_params, M = default_M)
    elif solver == "ode":
        return unified_model_ode(ppinterval, fs, seconds, basis_type, thetai, basis_params, solver)
    else:
        raise ValueError(f"Unsupported solver: {solver}")