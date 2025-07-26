import numpy as np
from numba import njit, prange
import math

@njit
def corrcoef_numba(x, y):
    """
    Calculate Pearson's Correlation Coefficient (r)
    :param x: vector 1
    :param y: vector 2
    :return: r
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cov = np.mean((x - x_mean) * (y - y_mean))
    std_x = np.std(x)
    std_y = np.std(y)
    return cov / (std_x * std_y + 1e-8)

@njit
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

@njit
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

    for i in range(len(arr)):
        acc = 0.0
        for j in range(-radius, radius + 1):
            idx = i + j
            if 0 <= idx < len(arr):
                acc += arr[idx] * kernel[j + radius]
        output[i] = acc
    return output

@njit
def gradient_1d(arr, dx=1.0):
    """
    Compute the gradient of a 1D array
    :param arr: input array/signal
    :param dx: delta between each point
    :return: gradient of arr
    """
    n = len(arr)
    grad = np.empty(n, dtype=arr.dtype)

    # Central differences
    for i in range(1, n - 1):
        grad[i] = (arr[i + 1] - arr[i - 1]) / (2 * dx)

    # Forward/backward difference at boundaries
    grad[0] = (arr[1] - arr[0]) / dx
    grad[-1] = (arr[-1] - arr[-2]) / dx
    return grad

@njit
def gamma_pdf(x, alpha, scale):
    if x <= 0.0:
        return 0.0
    return math.exp((alpha - 1.0)*math.log(x)
                    - x/scale
                    - math.lgamma(alpha)
                    - alpha*math.log(scale))

@njit
def norm_pdf(x, b):
    return math.exp(-0.5*(x/b)**2) / (math.sqrt(2*math.pi) * b)

@njit
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x/math.sqrt(2.0)))

@njit
def gamma_mean(alpha, scale, M):
    d0 = 2 * np.pi / M
    mean_val = 0.0
    for j in range(M):
        j0 = (j + 0.5) * d0
        mean_val += gamma_pdf(j0, alpha, scale)
    return mean_val * d0 / (2 * np.pi)

@njit
def skewed_gaussian_mean(b, skew, M):
    d0 = 2 * np.pi / M
    mean_val = 0.0
    for j in range(M):
        j0 = -np.pi + (j + 0.5) * d0
        mean_val += 2 * j0 * norm_pdf(j0, b) * norm_cdf(skew * j0 / b)
    return mean_val * d0 / (2 * np.pi)

@njit
def interp1d_lut(x, x_table, y_table):
    if x <= x_table[0]:
        return y_table[0]
    elif x >= x_table[-1]:
        return y_table[-1]
    for i in prange(len(x_table) - 1):
        if x_table[i] <= x <= x_table[i + 1]:
            dx = x_table[i + 1] - x_table[i]
            dy = y_table[i + 1] - y_table[i]
            return y_table[i] + (x - x_table[i]) * dy / dx
    return 0.0