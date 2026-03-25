#pragma once

#include <vector>
#include <complex>

namespace fastppg {

struct ScratchBuffer {
    int n_samples;
    int M;
    int L;
    int P;

    std::vector<double> rr;              // [n_samples]
    std::vector<double> theta;           // [n_samples]
    std::vector<double> z;               // [n_samples]
    std::vector<double> z_grid;          // [M]

    std::vector<double> x_table;         // [M] endpoint=False
    std::vector<double> x_table_ep;      // [M] endpoint=True

    std::vector<double> f_lut;           // [L * M]
    std::vector<double> G_lut;           // [L * M]
    std::vector<double> mean_vals;       // [L]
    std::vector<double> lut_vals;        // [L * M]
    std::vector<double> rk_traj;         // [n_samples * 3] or [n_samples * 4]  (ODE only)

    std::vector<std::complex<double>> fft_buf1;   // [M]
    std::vector<std::complex<double>> fft_buf2;   // [M]
    std::vector<double> fft_freq;                 // [M]
    std::vector<double> fft_real_buf;             // [M]

    std::vector<double> smooth1;         // [n_samples]
    std::vector<double> smooth2;         // [n_samples]
    std::vector<double> grad1;           // [n_samples]
    std::vector<double> grad2;           // [n_samples]
    std::vector<double> grad3;           // [n_samples]
    std::vector<double> grad4;           // [n_samples]
    std::vector<double> ranks1;          // [n_samples]
    std::vector<double> ranks2;          // [n_samples]

    ScratchBuffer() : n_samples(0), M(0), L(0), P(0) {}

    ScratchBuffer(int n_samples_, int M_, int L_, int P_)
        : n_samples(n_samples_), M(M_), L(L_), P(P_),
          rr(n_samples_), theta(n_samples_), z(n_samples_),
          z_grid(M_), x_table(M_), x_table_ep(M_),
          f_lut(L_ * M_), G_lut(L_ * M_),
          mean_vals(L_), lut_vals(L_ * M_),
          rk_traj(n_samples_ * 3),
          fft_buf1(M_), fft_buf2(M_), fft_freq(M_), fft_real_buf(M_),
          smooth1(n_samples_), smooth2(n_samples_),
          grad1(n_samples_), grad2(n_samples_),
          grad3(n_samples_), grad4(n_samples_),
          ranks1(n_samples_), ranks2(n_samples_)
    {
        double dx_nep = 2.0 * 3.14159265358979323846 / M_;
        for (int i = 0; i < M_; ++i) x_table[i] = i * dx_nep;

        double dx_ep = 2.0 * 3.14159265358979323846 / (M_ - 1);
        for (int i = 0; i < M_; ++i) x_table_ep[i] = i * dx_ep;
    }

    ScratchBuffer(const ScratchBuffer&) = delete;
    ScratchBuffer& operator=(const ScratchBuffer&) = delete;
    ScratchBuffer(ScratchBuffer&&) = default;
    ScratchBuffer& operator=(ScratchBuffer&&) = default;
};

}
