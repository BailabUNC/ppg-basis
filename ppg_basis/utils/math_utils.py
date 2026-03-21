import numpy as np
from numba import njit, prange
import math


_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)  # ≈ 0.3989422804
_INV_SQRT_2   = 1.0 / math.sqrt(2.0)             # ≈ 0.7071067812

@njit(cache=True)
def _wrap_pi(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

@njit(cache=True)
def _theta_to_index(theta, M):
    # map θ∈[-π,π] → index in [0, M)
    u = (theta + np.pi) * (M / (2.0*np.pi))
    j = int(np.floor(u)) % M
    frac = u - np.floor(u)
    return j, frac  # for linear interp between j and j+1

@njit(cache=True)
def corrcoef_numba(x, y):
    """
    Calculate Pearson's Correlation Coefficient (r)
    :param x: vector 1
    :param y: vector 2
    :return: r
    """
    n = x.shape[0]
    sx = 0.0; sy = 0.0
    for i in range(n):
        sx += x[i]; sy += y[i]
    mx = sx / n; my = sy / n
    cov = 0.0; vx = 0.0; vy = 0.0
    for i in range(n):
        dx = x[i] - mx
        dy = y[i] - my
        cov += dx * dy
        vx += dx * dx
        vy += dy * dy
    return cov / (math.sqrt(vx * vy) + 1e-8)

@njit(cache=True)
def gaussian_kernel1d(sigma, radius=3):
    """
    Generate a 1D Gaussian Kernel
    :param sigma: width of kernel
    :param radius: half-width of convolution window
    :return: Gaussian kernel to convolve
    """
    size = 2 * radius + 1
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma)**2)
    kernel /= kernel.sum()
    return kernel

@njit(cache=True)
def gaussian_filter1d_numba(arr, sigma):
    """
    Apply 1D Gaussian filter to array
    :param arr: input signal vector
    :param sigma: width of gaussian kernel
    :return: smoothed signal
    """
    radius = int(3 * sigma)
    kernel = gaussian_kernel1d(sigma, radius)
    output = np.zeros_like(arr)

    for i in prange(len(arr)):
        acc = 0.0
        for j in range(-radius, radius + 1):
            idx = i + j
            if 0 <= idx < len(arr):
                acc += arr[idx] * kernel[j + radius]
        output[i] = acc
    return output

@njit(cache=True)
def gradient_1d(arr, dx=1.0):
    """
    Compute the gradient of a 1D array
    :param arr: input array/signal
    :param dx: delta between each point
    :return: gradient of arr
    """
    n = len(arr)
    grad = np.empty(n, dtype=arr.dtype)
    inv_2dx = 1.0 / (2.0 * dx)
    inv_dx  = 1.0 / dx
    for i in prange(1, n - 1):
        grad[i] = (arr[i + 1] - arr[i - 1]) * inv_2dx
    grad[0] = (arr[1] - arr[0]) * inv_dx
    grad[-1] = (arr[-1] - arr[-2]) * inv_dx
    return grad

@njit(cache=True)
def gamma_pdf(x, alpha, scale):
    """
    Compute PDF of gamma func at a given point
    :param x: x value to compute PDF at
    :param alpha: shape parameter of gamma dist
    :param scale: scale parameter of gamma dist
    :return: Value of gamma PDF at x
    """
    if x <= 0.0:
        return 0.0
    return math.exp((alpha - 1.0)*math.log(x)
                    - x/scale
                    - math.lgamma(alpha)
                    - alpha*math.log(scale))

@njit(cache=True)
def gamma_pdf_precomp(x, alpha_m1, inv_scale, log_norm):
    """
    Gamma PDF with precomputed constants (avoids lgamma+log per call)
    """
    if x <= 0.0:
        return 0.0
    return math.exp(alpha_m1 * math.log(x) - x * inv_scale - log_norm)

@njit(cache=True)
def norm_pdf(x, b):
    """
    Compute PDF of normal distribution at given point
    :param x: x value to compute PDF at
    :param b: standard deviation
    :return: Value of normal PDF at x
    """
    inv_b = 1.0 / b
    return math.exp(-0.5 * (x * inv_b) ** 2) * inv_b * 0.3989422804014327

@njit(cache=True)
def norm_cdf(x):
    """
    Compute CDF of normal distribution at given point
    :param x: x value to compute CDF at
    :return: Value of normal CDF at x
    """
    return 0.5 * (1.0 + math.erf(x * 0.7071067811865476))

@njit(cache=True)
def gamma_mean(alpha, scale, M):
    """
    Approximate mean of gamma distribution over 2pi
    :param alpha: shape parameter of gamma dist
    :param scale: scale parameter of gamma dist
    :param M: number of integration steps
    :return: approximated mean of gamma dist over 2pi
    """
    d0 = 2 * np.pi / M
    alpha_m1 = alpha - 1.0
    inv_scale = 1.0 / scale
    log_norm = math.lgamma(alpha) + alpha * math.log(scale)
    mean_val = 0.0
    for j in range(M):
        j0 = (j + 0.5) * d0
        mean_val += gamma_pdf_precomp(j0, alpha_m1, inv_scale, log_norm)
    return mean_val * d0 / (2 * np.pi)


@njit(cache=True)
def skewed_gaussian_mean(b, skew, M):
    """
    Approximate mean of skewed Gaussian distribution over -pi to pi
    :param b: std of the Normal distribution
    :param skew: skewness parameter
    :param M: number of integration steps
    :return: approximated mean of skewed Gaussian distribution over -pi to pi
    """
    d0 = 2 * np.pi / M
    inv_b = 1.0 / max(b, 1e-6)
    skew_over_b = skew * inv_b
    mean_val = 0.0
    for j in range(M):
        j0 = -np.pi + (j + 0.5) * d0
        mean_val += skewed_gaussian_val(j0, inv_b, skew_over_b)
    return mean_val * d0 / (2 * np.pi)

@njit(cache=True)
def skewed_gaussian_val(x_centered, inv_b, skew_over_b):
    """
    Combined skewed-Gaussian derivative: 2·x·φ(x;b)·Φ(skew·x/b)
    with precomputed inv_b and skew_over_b
    """
    u = x_centered * inv_b
    pdf = math.exp(-0.5 * u * u) * inv_b * 0.3989422804014327
    cdf = 0.5 * (1.0 + math.erf(skew_over_b * x_centered * 0.7071067811865476))
    return 2.0 * x_centered * pdf * cdf

@njit(cache=True)
def interp1d_lut(x, x_table, y_table):
    """
    Perform 1D linear interpolation using lookup tables
    :param x: input value
    :param x_table: sorted x-values
    :param y_table: corresponding y-values
    :return: interpolated estimate of y = f(x)
    """
    if x <= x_table[0]:
        return y_table[0]
    elif x >= x_table[-1]:
        return y_table[-1]
    for i in range(len(x_table) - 1):
        if x_table[i] <= x <= x_table[i + 1]:
            dx = x_table[i + 1] - x_table[i]
            dy = y_table[i + 1] - y_table[i]
            return y_table[i] + (x - x_table[i]) * dy / dx
    return 0.0

@njit(cache=True)
def _interp1d_lut_scalar(x, x_table, y_table):
    """
    linear interpolation for scalars. Assumes x_table is uniform over [0, 2π].
    :param x: input value
    :param x_table: sorted x-values
    :param y_table: corresponding y-values
    :return: interpolated estimate of y = f(x)
    """
    M = x_table.shape[0]
    dx = x_table[1] - x_table[0]
    inv_dx = 1.0 / dx
    if x < 0.0:
        x = 0.0
    elif x > 2.0*np.pi:
        x = 2.0*np.pi
    idx = int(x * inv_dx)
    if idx >= M-1:
        idx = M-2
    x0 = idx * dx
    t = (x - x0) * inv_dx
    return (1.0 - t) * y_table[idx] + t * y_table[idx + 1]

@njit(cache=True)
def _interp_uniform_table(theta, z_grid):
    # scalar interpolation of a uniform 2π-periodic table
    M = z_grid.shape[0]
    j, frac = _theta_to_index(theta, M)
    j2 = (j + 1) % M
    return (1.0 - frac) * z_grid[j] + frac * z_grid[j2]

@njit
def rank_array(arr):
    """
    Rank input array
    :param arr: array
    """
    n = len(arr)
    ranks = np.empty(n, dtype=np.float64)
    sorted_idx = np.argsort(arr)
    ranks[sorted_idx[0]] = 0
    for i in range(1,n):
        if arr[sorted_idx[i]] == arr[sorted_idx[i-1]]:
            ranks[sorted_idx[i]] = ranks[sorted_idx[i-1]]
        else:
            ranks[sorted_idx[i]] = i
    return ranks
