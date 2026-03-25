#pragma once

#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>

namespace fastppg { namespace fft {

using cdouble = std::complex<double>;
static constexpr double PI = 3.14159265358979323846;

// Utilities
inline bool is_power_of_2(int n) { return n > 0 && (n & (n - 1)) == 0; }

inline int next_power_of_2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Bit reversal
inline void bit_reverse(cdouble* data, int n) {
    int j = 0;
    for (int i = 0; i < n; ++i) {
        if (j > i) std::swap(data[i], data[j]);
        int m = n >> 1;
        while (m >= 1 && j >= m) { j -= m; m >>= 1; }
        j += m;
    }
}

// Cooley-Tukey
inline void ct_fft_inplace(cdouble* data, int n, int sign) {
    bit_reverse(data, n);
    for (int len = 2; len <= n; len <<= 1) {
        double angle = sign * (-2.0 * PI / len);
        cdouble wn(std::cos(angle), std::sin(angle));
        for (int i = 0; i < n; i += len) {
            cdouble w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                cdouble u = data[i + j];
                cdouble t = w * data[i + j + len / 2];
                data[i + j]             = u + t;
                data[i + j + len / 2]   = u - t;
                w *= wn;
            }
        }
    }
}

// Bluestein chirp-z
inline void bluestein_fft(const cdouble* in, int n, cdouble* out, int sign) {
    int m = next_power_of_2(2 * n - 1);  // convolution length

    // chirp sequence
    std::vector<cdouble> chirp(n);
    for (int k = 0; k < n; ++k) {
        double angle = sign * (-PI * (double)(k * k) / n);
        chirp[k] = cdouble(std::cos(angle), std::sin(angle));
    }

    std::vector<cdouble> a(m, cdouble(0.0));
    for (int k = 0; k < n; ++k)
        a[k] = in[k] * std::conj(chirp[k]);

    std::vector<cdouble> b(m, cdouble(0.0));
    for (int k = 0; k < n; ++k)
        b[k] = chirp[k];
    for (int k = 1; k < n; ++k)
        b[m - k] = chirp[k];

    // convolve via FFT
    ct_fft_inplace(a.data(), m, 1);
    ct_fft_inplace(b.data(), m, 1);
    for (int k = 0; k < m; ++k)
        a[k] *= b[k];
    ct_fft_inplace(a.data(), m, -1);
    double inv_m = 1.0 / m;

    // extract result
    for (int k = 0; k < n; ++k)
        out[k] = a[k] * inv_m * std::conj(chirp[k]);
}

// Complex API
inline void c2c(cdouble* data, int n, int sign) {
    if (is_power_of_2(n)) {
        ct_fft_inplace(data, n, sign);
        if (sign == -1) {
            double inv_n = 1.0 / n;
            for (int i = 0; i < n; ++i) data[i] *= inv_n;
        }
    } else {
        std::vector<cdouble> tmp(n);
        bluestein_fft(data, n, tmp.data(), sign);
        if (sign == -1) {
            double inv_n = 1.0 / n;
            for (int i = 0; i < n; ++i) data[i] = tmp[i] * inv_n;
        } else {
            std::copy(tmp.begin(), tmp.end(), data);
        }
    }
}

inline void fft_forward(const double* in, int M, cdouble* out) {
    for (int i = 0; i < M; ++i) out[i] = cdouble(in[i], 0.0);
    c2c(out, M, 1);
}

inline void fft_inverse_real(cdouble* data, int M, double* out) {
    c2c(data, M, -1);
    for (int i = 0; i < M; ++i) out[i] = data[i].real();
}

inline void fftfreq_scaled(int M, double* out) {
    for (int i = 0; i <= M / 2 - 1; ++i)
        out[i] = (double)i;
    for (int i = M / 2; i < M; ++i)
        out[i] = (double)(i - M);
}

}}
