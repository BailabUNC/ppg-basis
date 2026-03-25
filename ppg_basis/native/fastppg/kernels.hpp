#pragma once

#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <vector>
#include <limits>

namespace fastppg { namespace kernels {

// Constants
static constexpr double PI        = 3.14159265358979323846;
static constexpr double TWO_PI    = 6.28318530717958647692;
static constexpr double INV_SQRT_2PI = 0.3989422804014327;   // 1/sqrt(2π)
static constexpr double INV_SQRT_2   = 0.7071067811865476;   // 1/sqrt(2)

// Angle wrap
inline double wrap_pi(double x) {
    x = std::fmod(x + PI, TWO_PI);
    if (x < 0.0) x += TWO_PI;
    return x - PI;
}

// index mapping on uniform [−π, π] to [0, M]
inline void theta_to_index(double theta, int M, int& j, double& frac) {
    double u = (theta + PI) * (M / TWO_PI);
    double fu = std::floor(u);
    j = ((int)fu) % M;
    if (j < 0) j += M;
    frac = u - fu;
}

// R
inline double corrcoef(const double* x, const double* y, int n) {
    double sx = 0.0, sy = 0.0;
    for (int i = 0; i < n; ++i) { sx += x[i]; sy += y[i]; }
    double mx = sx / n, my = sy / n;
    double cov = 0.0, vx = 0.0, vy = 0.0;
    for (int i = 0; i < n; ++i) {
        double dx = x[i] - mx;
        double dy = y[i] - my;
        cov += dx * dy;
        vx  += dx * dx;
        vy  += dy * dy;
    }
    return cov / (std::sqrt(vx * vy) + 1e-8);
}

// Gaussian kernel
inline void gaussian_kernel1d(double sigma, int radius, double* out) {
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        double v = std::exp(-0.5 * (i / sigma) * (i / sigma));
        out[i + radius] = v;
        sum += v;
    }
    double inv_sum = 1.0 / sum;
    for (int i = 0; i < 2 * radius + 1; ++i) out[i] *= inv_sum;
}

inline void gaussian_filter1d(const double* arr, int n, double sigma,
                              double* out) {
    int radius = (int)(3.0 * sigma);
    if (radius < 1) radius = 1;
    int ksize = 2 * radius + 1;
    std::vector<double> kernel(ksize);
    gaussian_kernel1d(sigma, radius, kernel.data());
    for (int i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int j = -radius; j <= radius; ++j) {
            int idx = i + j;
            if (idx >= 0 && idx < n)
                acc += arr[idx] * kernel[j + radius];
        }
        out[i] = acc;
    }
}

inline void gradient_1d(const double* arr, int n, double dx, double* out) {
    double inv_2dx = 1.0 / (2.0 * dx);
    double inv_dx  = 1.0 / dx;
    out[0] = (arr[1] - arr[0]) * inv_dx;
    for (int i = 1; i < n - 1; ++i)
        out[i] = (arr[i + 1] - arr[i - 1]) * inv_2dx;
    out[n - 1] = (arr[n - 1] - arr[n - 2]) * inv_dx;
}

inline double gamma_pdf(double x, double alpha, double scale) {
    if (x <= 0.0) return 0.0;
    return std::exp((alpha - 1.0) * std::log(x)
                    - x / scale
                    - std::lgamma(alpha)
                    - alpha * std::log(scale));
}

inline double gamma_pdf_precomp(double x, double alpha_m1, double inv_scale,
                                double log_norm) {
    if (x <= 0.0) return 0.0;
    return std::exp(alpha_m1 * std::log(x) - x * inv_scale - log_norm);
}

inline double norm_pdf(double x, double b) {
    double inv_b = 1.0 / b;
    return std::exp(-0.5 * (x * inv_b) * (x * inv_b)) * inv_b * INV_SQRT_2PI;
}

inline double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x * INV_SQRT_2));
}

inline double skewed_gaussian_val(double x_centered, double inv_b,
                                  double skew_over_b) {
    double u = x_centered * inv_b;
    double pdf = std::exp(-0.5 * u * u) * inv_b * INV_SQRT_2PI;
    double cdf = 0.5 * (1.0 + std::erf(skew_over_b * x_centered * INV_SQRT_2));
    return 2.0 * x_centered * pdf * cdf;
}

inline double gamma_mean(double alpha, double scale, int M) {
    double d0 = TWO_PI / M;
    double alpha_m1 = alpha - 1.0;
    double inv_scale = 1.0 / scale;
    double log_norm = std::lgamma(alpha) + alpha * std::log(scale);
    double mean_val = 0.0;
    for (int j = 0; j < M; ++j) {
        double j0 = (j + 0.5) * d0;
        mean_val += gamma_pdf_precomp(j0, alpha_m1, inv_scale, log_norm);
    }
    return mean_val * d0 / TWO_PI;
}

inline double skewed_gaussian_mean(double b, double skew, int M) {
    double d0 = TWO_PI / M;
    double bb = std::max(b, 1e-6);
    double inv_b = 1.0 / bb;
    double skew_over_b = skew * inv_b;
    double mean_val = 0.0;
    for (int j = 0; j < M; ++j) {
        double j0 = -PI + (j + 0.5) * d0;
        mean_val += skewed_gaussian_val(j0, inv_b, skew_over_b);
    }
    return mean_val * d0 / TWO_PI;
}

inline double interp1d_lut(double x, const double* x_table,
                           const double* y_table, int M) {
    if (x <= x_table[0]) return y_table[0];
    if (x >= x_table[M - 1]) return y_table[M - 1];
    double dx = x_table[1] - x_table[0];
    int idx = (int)((x - x_table[0]) / dx);
    if (idx >= M - 1) idx = M - 2;
    double t = (x - x_table[idx]) / dx;
    return (1.0 - t) * y_table[idx] + t * y_table[idx + 1];
}

inline double interp1d_lut_scalar(double x, const double* x_table,
                                  const double* y_table, int M) {
    double dx = x_table[1] - x_table[0];
    double inv_dx = 1.0 / dx;
    if (x < 0.0) x = 0.0;
    else if (x > TWO_PI) x = TWO_PI;
    int idx = (int)(x * inv_dx);
    if (idx >= M - 1) idx = M - 2;
    double x0 = idx * dx;
    double t = (x - x0) * inv_dx;
    return (1.0 - t) * y_table[idx] + t * y_table[idx + 1];
}

// Circular interpolation
inline double interp_uniform_table(double theta, const double* z_grid, int M) {
    int j; double frac;
    theta_to_index(theta, M, j, frac);
    int j2 = (j + 1) % M;
    return (1.0 - frac) * z_grid[j] + frac * z_grid[j2];
}

inline void rank_array(const double* arr, int n, double* ranks) {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [arr](int a, int b) { return arr[a] < arr[b]; });
    ranks[idx[0]] = 0.0;
    for (int i = 1; i < n; ++i) {
        if (arr[idx[i]] == arr[idx[i - 1]])
            ranks[idx[i]] = ranks[idx[i - 1]];
        else
            ranks[idx[i]] = (double)i;
    }
}

inline void detrend_linear(double* arr, int n) {
    double mean_t = (n - 1.0) / 2.0;
    double sum_y = 0.0;
    for (int i = 0; i < n; ++i) sum_y += arr[i];
    double mean_y = sum_y / n;

    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        double dt = i - mean_t;
        num += dt * (arr[i] - mean_y);
        den += dt * dt;
    }
    double b = (den > 1e-30) ? num / den : 0.0;
    double a = mean_y - b * mean_t;
    for (int i = 0; i < n; ++i)
        arr[i] -= (a + b * i);
}

inline void postprocess_z(double* z, int n) {
    // nan_to_num
    for (int i = 0; i < n; ++i) {
        if (std::isnan(z[i]) || std::isinf(z[i])) z[i] = 0.0;
    }
    // linear detrend
    detrend_linear(z, n);
    // zero-mean
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += z[i];
    mean /= n;
    for (int i = 0; i < n; ++i) z[i] -= mean;
    // min-max normalize
    double mn = z[0], mx = z[0];
    for (int i = 1; i < n; ++i) {
        if (z[i] < mn) mn = z[i];
        if (z[i] > mx) mx = z[i];
    }
    double scale = 1.0 / (mx - mn + 1e-8);
    for (int i = 0; i < n; ++i) z[i] = (z[i] - mn) * scale;
}

}}
