import numpy as np
from numba import njit, prange

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