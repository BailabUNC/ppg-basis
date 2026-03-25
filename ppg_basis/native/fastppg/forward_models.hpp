#pragma once

#include "kernels.hpp"
#include "fft_utils.hpp"
#include "scratch_buffer.hpp"
#include <cstring>
#include <cmath>
#include <string>
#include <stdexcept>

namespace fastppg { namespace models {

using namespace fastppg::kernels;
using cdouble = std::complex<double>;

enum class BasisType : uint8_t { GAUSSIAN, GAMMA, SKEWED_GAUSSIAN };

inline BasisType parse_basis_type(const std::string& s) {
    if (s == "gaussian")         return BasisType::GAUSSIAN;
    if (s == "gamma")            return BasisType::GAMMA;
    if (s == "skewed-gaussian")  return BasisType::SKEWED_GAUSSIAN;
    throw std::runtime_error("Unsupported basis type: " + s);
}

inline int params_per_basis(BasisType bt) {
    switch (bt) {
        case BasisType::GAUSSIAN:        return 2;
        case BasisType::GAMMA:           return 3;
        case BasisType::SKEWED_GAUSSIAN: return 3;
    }
    return 2;
}

enum class SolverType : uint8_t { BASIS, TEMPLATE, FFT, RK3, RK4 };

inline SolverType parse_solver(const std::string& s) {
    if (s == "basis")    return SolverType::BASIS;
    if (s == "template") return SolverType::TEMPLATE;
    if (s == "fft")      return SolverType::FFT;
    if (s == "rk3")      return SolverType::RK3;
    if (s == "rk4")      return SolverType::RK4;
    throw std::runtime_error("Unsupported solver: " + s);
}

inline void phase_from_rr(const double* ppinterval, int n_pp, double fs,
                          int n_samples, double* rr_out, double* theta_out) {
    int total = 0, idx = 0;
    while (total < n_samples) {
        double intv = ppinterval[idx % n_pp];
        int num = (int)std::ceil(fs * intv);
        int end = std::min(total + num, n_samples);
        for (int i = total; i < end; ++i) rr_out[i] = intv;
        total = end;
        idx++;
    }
    double dt = 1.0 / fs;
    theta_out[0] = -PI;
    for (int k = 1; k < n_samples; ++k) {
        double omega = TWO_PI / rr_out[k - 1];
        theta_out[k] = wrap_pi(theta_out[k - 1] + omega * dt);
    }
}

inline void precompute_f_and_G(const double* basis_params, int L, int P,
                               BasisType bt, int M,
                               const double* x_table_ep, double dx_ep,
                               double* f_lut, double* G_lut) {
    std::vector<double> mean_vals(L, 0.0);

    for (int i = 0; i < L; ++i) {
        double* fi = f_lut + i * M;
        const double* pi = basis_params + i * P;

        if (bt == BasisType::GAUSSIAN) {
            double b = pi[1];
            double bb = std::max(b, 1e-6);
            double inv2b2 = 1.0 / (2.0 * bb * bb);
            for (int j = 0; j < M; ++j) {
                double x = x_table_ep[j] - PI;
                fi[j] = x * std::exp(-(x * x) * inv2b2);
            }
            mean_vals[i] = 0.0;

        } else if (bt == BasisType::GAMMA) {
            double alpha = pi[1], scale = pi[2];
            double alpha_m1 = alpha - 1.0;
            double safe_scale = std::max(scale, 1e-10);
            double inv_scale = 1.0 / safe_scale;
            double log_norm = std::lgamma(alpha) + alpha * std::log(safe_scale);
            for (int j = 0; j < M; ++j)
                fi[j] = gamma_pdf_precomp(x_table_ep[j], alpha_m1, inv_scale, log_norm);
            // trapezoid mean
            double s = 0.5 * (fi[0] + fi[M - 1]);
            for (int j = 1; j < M - 1; ++j) s += fi[j];
            mean_vals[i] = s * dx_ep / TWO_PI;

        } else { // skewed_gaussian case
            double b = pi[1], skew = pi[2];
            double bb = std::max(b, 1e-6);
            double inv_b = 1.0 / bb;
            double sob = skew * inv_b;
            for (int j = 0; j < M; ++j) {
                double x = x_table_ep[j] - PI;
                fi[j] = skewed_gaussian_val(x, inv_b, sob);
            }
            double s = 0.5 * (fi[0] + fi[M - 1]);
            for (int j = 1; j < M - 1; ++j) s += fi[j];
            mean_vals[i] = s * dx_ep / TWO_PI;
        }
    }

    // zero-mean so G is periodic
    for (int i = 0; i < L; ++i) {
        double* fi = f_lut + i * M;
        for (int j = 0; j < M; ++j) fi[j] -= mean_vals[i];
    }

    double half_dx = 0.5 * dx_ep;
    for (int i = 0; i < L; ++i) {
        const double* fi = f_lut + i * M;
        double* Gi = G_lut + i * M;
        double acc = 0.0;
        Gi[0] = 0.0;
        for (int j = 1; j < M; ++j) {
            acc += (fi[j - 1] + fi[j]) * half_dx;
            Gi[j] = acc;
        }
        // remove residual slope for periodicity
        double slope = Gi[M - 1] / TWO_PI;
        for (int j = 0; j < M; ++j)
            Gi[j] -= slope * x_table_ep[j];
    }
}

// Sample z_grid at phase angles
inline void sample_template(const double* theta, int n,
                            const double* z_grid, int M, double* out) {
    for (int k = 0; k < n; ++k)
        out[k] = interp_uniform_table(theta[k], z_grid, M);
}

inline void synthesize_gaussian_core(const double* theta, int n,
                                     const double* thetai,
                                     const double* basis_params,
                                     int L, int P, double* z) {
    std::memset(z, 0, n * sizeof(double));
    for (int i = 0; i < L; ++i) {
        double a = basis_params[i * P + 0];
        double b = std::max(basis_params[i * P + 1], 1e-6);
        double inv2b2 = 1.0 / (2.0 * b * b);
        double amp = a * b * b;
        for (int k = 0; k < n; ++k) {
            double diff = std::fmod(theta[k] - thetai[i] + PI, TWO_PI);
            if (diff < 0.0) diff += TWO_PI;
            diff -= PI;
            z[k] += amp * std::exp(-(diff * diff) * inv2b2);
        }
    }
}

inline void synthesize_basis_core(const double* theta, int n,
                                  const double* thetai,
                                  const double* basis_params,
                                  int L, int P, int M,
                                  const double* x_table_ep,
                                  const double* G_lut, double* z) {
    std::memset(z, 0, n * sizeof(double));
    for (int i = 0; i < L; ++i) {
        double a = basis_params[i * P + 0];
        const double* Gi = G_lut + i * M;
        for (int k = 0; k < n; ++k) {
            double x = wrap_pi(theta[k] - thetai[i]) + PI;
            double val = interp1d_lut_scalar(x, x_table_ep, Gi, M);
            z[k] -= a * val;
        }
    }
}

inline void forward_basis(const double* ppinterval, int n_pp, double fs,
                          double seconds, BasisType bt,
                          const double* thetai, const double* basis_params,
                          int L, int P, int M, ScratchBuffer& buf) {
    int n = (int)std::ceil(seconds * fs);
    phase_from_rr(ppinterval, n_pp, fs, n, buf.rr.data(), buf.theta.data());

    if (bt == BasisType::GAUSSIAN) {
        synthesize_gaussian_core(buf.theta.data(), n, thetai, basis_params,
                                 L, P, buf.z.data());
    } else {
        double dx_ep = buf.x_table_ep[1] - buf.x_table_ep[0];
        precompute_f_and_G(basis_params, L, P, bt, M,
                           buf.x_table_ep.data(), dx_ep,
                           buf.f_lut.data(), buf.G_lut.data());
        synthesize_basis_core(buf.theta.data(), n, thetai, basis_params,
                              L, P, M, buf.x_table_ep.data(),
                              buf.G_lut.data(), buf.z.data());
    }
    postprocess_z(buf.z.data(), n);
}

inline void build_phase_template_gaussian(const double* thetai,
                                          const double* basis_params,
                                          int L, int P, int M,
                                          double* z_grid) {
    double d_phi = TWO_PI / M;
    std::memset(z_grid, 0, M * sizeof(double));
    for (int i = 0; i < L; ++i) {
        double a = basis_params[i * P + 0];
        double b = std::max(basis_params[i * P + 1], 1e-6);
        double amp = a * b * b;
        for (int j = 0; j < M; ++j) {
            double phi = j * d_phi;
            double raw = phi - (thetai[i] + PI);
            double diff = std::fmod(raw, TWO_PI);
            if (diff < 0.0) diff += TWO_PI;
            diff -= PI;
            z_grid[j] += amp * std::exp(-0.5 * (diff / b) * (diff / b));
        }
    }
}

inline void build_phase_template_generic(const double* thetai,
                                         const double* basis_params,
                                         int L, int P, BasisType bt, int M,
                                         const double* x_table_ep, double dx_ep,
                                         double* f_lut, double* G_lut,
                                         cdouble* fft1, cdouble* fft2,
                                         double* freq,
                                         double* z_grid) {
    precompute_f_and_G(basis_params, L, P, bt, M, x_table_ep, dx_ep,
                       f_lut, G_lut);

    fft::fftfreq_scaled(M, freq);
    std::memset(z_grid, 0, M * sizeof(double));

    for (int i = 0; i < L; ++i) {
        double a = basis_params[i * P + 0];
        double d = (thetai[i] + PI) * M / TWO_PI;

        // FFT of G_lut[i]
        const double* Gi = G_lut + i * M;
        for (int j = 0; j < M; ++j) fft1[j] = cdouble(Gi[j], 0.0);
        fft::c2c(fft1, M, 1);

        // apply phase shift
        for (int k = 0; k < M; ++k) {
            double phase = -TWO_PI * freq[k] * d / M;
            fft1[k] *= cdouble(std::cos(phase), std::sin(phase));
        }

        // IFFT
        fft::c2c(fft1, M, -1);
        for (int j = 0; j < M; ++j)
            z_grid[j] -= a * fft1[j].real();
    }
}

inline void forward_template(const double* ppinterval, int n_pp, double fs,
                             double seconds, BasisType bt,
                             const double* thetai, const double* basis_params,
                             int L, int P, int M, ScratchBuffer& buf) {
    int n = (int)std::ceil(seconds * fs);
    phase_from_rr(ppinterval, n_pp, fs, n, buf.rr.data(), buf.theta.data());

    if (bt == BasisType::GAUSSIAN) {
        build_phase_template_gaussian(thetai, basis_params, L, P, M,
                                      buf.z_grid.data());
    } else {
        double dx_ep = buf.x_table_ep[1] - buf.x_table_ep[0];
        build_phase_template_generic(thetai, basis_params, L, P, bt, M,
                                     buf.x_table_ep.data(), dx_ep,
                                     buf.f_lut.data(), buf.G_lut.data(),
                                     buf.fft_buf1.data(), buf.fft_buf2.data(),
                                     buf.fft_freq.data(),
                                     buf.z_grid.data());
    }
    sample_template(buf.theta.data(), n, buf.z_grid.data(), M, buf.z.data());
    postprocess_z(buf.z.data(), n);
}

inline void tabulate_zero_mean_derivative(BasisType bt,
                                          const double* basis_params,
                                          int L, int P, int M,
                                          double* g) {
    double d_phi = TWO_PI / M;
    std::memset(g, 0, M * sizeof(double));

    if (bt == BasisType::GAUSSIAN) {
        for (int i = 0; i < L; ++i) {
            double b = std::max(basis_params[i * P + 1], 1e-6);
            for (int j = 0; j < M; ++j) {
                double x = j * d_phi - PI;
                g[j] += x * std::exp(-0.5 * (x / b) * (x / b));
            }
        }
    } else if (bt == BasisType::GAMMA) {
        for (int i = 0; i < L; ++i) {
            double alpha = basis_params[i * P + 1];
            double scale = basis_params[i * P + 2];
            double safe_scale = std::max(scale, 1e-10);
            for (int j = 0; j < M; ++j) {
                double phi = j * d_phi;
                if (phi > 0.0) {
                    double lp = (alpha - 1.0) * std::log(phi) - phi / safe_scale
                                - std::lgamma(alpha) - alpha * std::log(safe_scale);
                    g[j] += std::exp(lp);
                }
            }
        }
    } else { // skewed_guassian case
        for (int i = 0; i < L; ++i) {
            double b = std::max(basis_params[i * P + 1], 1e-6);
            double skew = basis_params[i * P + 2];
            for (int j = 0; j < M; ++j) {
                double x = j * d_phi - PI;
                double pdf_val = std::exp(-0.5 * (x / b) * (x / b))
                                 / (std::sqrt(TWO_PI) * b);
                double cdf_val = 0.5 * (1.0 + std::erf(skew * x / (b * std::sqrt(2.0))));
                g[j] += 2.0 * x * pdf_val * cdf_val;
            }
        }
    }
    double inv_L = 1.0 / std::max(L, 1);
    double mean = 0.0;
    for (int j = 0; j < M; ++j) { g[j] *= inv_L; mean += g[j]; }
    mean /= M;
    for (int j = 0; j < M; ++j) g[j] -= mean;
}

inline void primitive_coeffs_from_derivative_fft(const cdouble* F, int M,
                                                 const double* freq,
                                                 cdouble* G) {
    G[0] = cdouble(0.0, 0.0);
    for (int k = 1; k < M; ++k) {
        if (freq[k] != 0.0)
            G[k] = F[k] / (cdouble(0.0, 1.0) * freq[k]);
        else
            G[k] = cdouble(0.0, 0.0);
    }
}

inline void impulse_train_coeffs(const double* thetai,
                                 const double* basis_params,
                                 int L, int P, int M,
                                 const double* freq,
                                 cdouble* S) {
    std::memset(S, 0, M * sizeof(cdouble));
    for (int i = 0; i < L; ++i) {
        double a = basis_params[i * P + 0];
        for (int k = 0; k < M; ++k) {
            double angle = -thetai[i] * freq[k];
            S[k] += a * cdouble(std::cos(angle), std::sin(angle));
        }
    }
}

inline void forward_fft(const double* ppinterval, int n_pp, double fs,
                        double seconds, BasisType bt,
                        const double* thetai, const double* basis_params,
                        int L, int P, int M, ScratchBuffer& buf) {
    int n = (int)std::ceil(seconds * fs);
    phase_from_rr(ppinterval, n_pp, fs, n, buf.rr.data(), buf.theta.data());

    // tabulate g on [0, 2π)
    tabulate_zero_mean_derivative(bt, basis_params, L, P, M,
                                  buf.fft_real_buf.data());

    // FFT(g)
    fft::fftfreq_scaled(M, buf.fft_freq.data());
    fft::fft_forward(buf.fft_real_buf.data(), M, buf.fft_buf1.data());

    // G_k = F_k / (ik)
    primitive_coeffs_from_derivative_fft(buf.fft_buf1.data(), M,
                                         buf.fft_freq.data(),
                                         buf.fft_buf2.data());

    // S_k = impulse train
    impulse_train_coeffs(thetai, basis_params, L, P, M,
                         buf.fft_freq.data(), buf.fft_buf1.data());

    // Z_k = -G_k * S_k
    for (int k = 0; k < M; ++k)
        buf.fft_buf1[k] = -buf.fft_buf2[k] * buf.fft_buf1[k];

    // IFFT → z_grid
    fft::fft_inverse_real(buf.fft_buf1.data(), M, buf.z_grid.data());

    sample_template(buf.theta.data(), n, buf.z_grid.data(), M, buf.z.data());
    postprocess_z(buf.z.data(), n);
}

// Precompute mean and LUT values for ODE basis evaluation
inline void precompute_mean_basis_values(const double* basis_params, int L, int P,
                                         BasisType bt, const double* x_table_ep,
                                         int M, double* mean_vals, double* lut_vals) {
    double dx = x_table_ep[1] - x_table_ep[0];
    for (int i = 0; i < L; ++i) {
        double* lv = lut_vals + i * M;
        const double* pi = basis_params + i * P;

        if (bt == BasisType::GAMMA) {
            double alpha = pi[1], scale = pi[2];
            double alpha_m1 = alpha - 1.0;
            double safe_s = std::max(scale, 1e-10);
            double inv_s = 1.0 / safe_s;
            double log_n = std::lgamma(alpha) + alpha * std::log(safe_s);
            for (int j = 0; j < M; ++j)
                lv[j] = gamma_pdf_precomp(x_table_ep[j], alpha_m1, inv_s, log_n);
            double s = 0.5 * (lv[0] + lv[M - 1]);
            for (int j = 1; j < M - 1; ++j) s += lv[j];
            mean_vals[i] = s * dx / TWO_PI;

        } else if (bt == BasisType::SKEWED_GAUSSIAN) {
            double b = pi[1], skew = pi[2];
            double bb = std::max(b, 1e-6);
            double inv_b = 1.0 / bb;
            double sob = skew * inv_b;
            for (int j = 0; j < M; ++j) {
                double x = x_table_ep[j] - PI;
                lv[j] = skewed_gaussian_val(x, inv_b, sob);
            }
            double s = 0.5 * (lv[0] + lv[M - 1]);
            for (int j = 1; j < M - 1; ++j) s += lv[j];
            mean_vals[i] = s * dx / TWO_PI;

        } else { // gaussian case
            mean_vals[i] = 0.0;
            std::memset(lv, 0, M * sizeof(double));
        }
    }
}

// RK integration
inline void rk_integration(bool rk4_flag,
                           const double* tspan, int n_steps,
                           const double* rr, int n_rr, double fs,
                           const double* thetai,
                           const double* basis_params, int L, int P,
                           BasisType bt, const double* mean_vals,
                           const double* x_table_ep, const double* lut_vals,
                           int M, double* traj) {
    double dt = tspan[1] - tspan[0];
    traj[0] = -1.0; traj[1] = 0.0; traj[2] = 0.0;

    double dx_table = x_table_ep[1] - x_table_ep[0];
    double x_table_max = x_table_ep[M - 1];

    auto gen_eq = [&](double t, const double* pt, double* dy) {
        int ip = std::min((int)std::floor(t * fs), n_rr - 1);
        if (ip < 0) ip = 0;
        double w = TWO_PI / rr[ip];
        double x = pt[0], y = pt[1];
        dy[0] = -w * y;
        dy[1] =  w * x;
        dy[2] = 0.0;
        double theta = std::atan2(y, x);
        for (int i = 0; i < L; ++i) {
            double diff = std::fmod(theta - thetai[i] + PI, TWO_PI);
            if (diff < 0.0) diff += TWO_PI;
            diff -= PI;
            double f;
            if (bt == BasisType::GAUSSIAN) {
                double a = basis_params[i * P + 0];
                double b = basis_params[i * P + 1];
                double b_sq = std::max(b * b, 1e-6);
                f = a * diff * std::exp(-(diff * diff) / (2.0 * b_sq));
            } else {
                double xval = diff + PI;
                if (xval < 0.0) xval = 0.0;
                else if (xval > x_table_max) xval = x_table_max;
                int idx = (int)(xval / dx_table);
                if (idx >= M - 1) idx = M - 2;
                double frac = (xval - idx * dx_table) / dx_table;
                const double* lv = lut_vals + i * M;
                double f_interp = (1.0 - frac) * lv[idx] + frac * lv[idx + 1];
                f = (f_interp - mean_vals[i]) * basis_params[i * P + 0];
            }
            dy[2] -= f * w;
        }
    };

    double k1[3], k2[3], k3[3], k4[3], tmp[3];

    for (int step = 1; step < n_steps; ++step) {
        double t = tspan[step - 1];
        const double* y_prev = traj + (step - 1) * 3;
        double* y_next = traj + step * 3;

        gen_eq(t, y_prev, k1);

        if (rk4_flag) {
            // RK4
            for (int d = 0; d < 3; ++d) tmp[d] = y_prev[d] + dt * k1[d] / 2.0;
            gen_eq(t + dt / 2.0, tmp, k2);
            for (int d = 0; d < 3; ++d) tmp[d] = y_prev[d] + dt * k2[d] / 2.0;
            gen_eq(t + dt / 2.0, tmp, k3);
            for (int d = 0; d < 3; ++d) tmp[d] = y_prev[d] + dt * k3[d];
            gen_eq(t + dt, tmp, k4);
            for (int d = 0; d < 3; ++d)
                y_next[d] = y_prev[d] + (dt / 6.0) * (k1[d] + 2*k2[d] + 2*k3[d] + k4[d]);
        } else {
            // RK3
            for (int d = 0; d < 3; ++d) tmp[d] = y_prev[d] + dt * k1[d] / 2.0;
            gen_eq(t + dt / 2.0, tmp, k2);
            for (int d = 0; d < 3; ++d) tmp[d] = y_prev[d] - dt * k1[d] + 2.0 * dt * k2[d];
            gen_eq(t + dt, tmp, k3);
            for (int d = 0; d < 3; ++d)
                y_next[d] = y_prev[d] + (dt / 6.0) * (k1[d] + 4*k2[d] + k3[d]);
        }
    }
}

inline void forward_ode(const double* ppinterval, int n_pp, double fs,
                        double seconds, BasisType bt, SolverType ode_solver,
                        const double* thetai, const double* basis_params,
                        int L, int P, int M, ScratchBuffer& buf) {
    int n = (int)std::ceil(seconds * fs);
    double dt = 1.0 / fs;

    {
        int total = 0, idx = 0;
        while (total < n) {
            double intv = ppinterval[idx % n_pp];
            int num = (int)std::ceil(fs * intv);
            int end = std::min(total + num, n);
            for (int i = total; i < end; ++i) buf.rr[i] = intv;
            total = end;
            idx++;
        }
    }

    // tspan
    std::vector<double> tspan(n);
    for (int i = 0; i < n; ++i) tspan[i] = i * dt;

    precompute_mean_basis_values(basis_params, L, P, bt,
                                buf.x_table_ep.data(), M,
                                buf.mean_vals.data(), buf.lut_vals.data());

    rk_integration(ode_solver == SolverType::RK4,
                   tspan.data(), n,
                   buf.rr.data(), n, fs,
                   thetai, basis_params, L, P, bt,
                   buf.mean_vals.data(), buf.x_table_ep.data(),
                   buf.lut_vals.data(), M,
                   buf.rk_traj.data());

    for (int i = 0; i < n; ++i)
        buf.z[i] = buf.rk_traj[i * 3 + 2];

    postprocess_z(buf.z.data(), n);
}

inline void forward_model(const double* ppinterval, int n_pp, double fs,
                          double seconds, BasisType bt, SolverType solver,
                          const double* thetai, const double* basis_params,
                          int L, int P, int M, ScratchBuffer& buf) {
    switch (solver) {
        case SolverType::BASIS:
            forward_basis(ppinterval, n_pp, fs, seconds, bt,
                          thetai, basis_params, L, P, M, buf);
            break;
        case SolverType::TEMPLATE:
            forward_template(ppinterval, n_pp, fs, seconds, bt,
                             thetai, basis_params, L, P, M, buf);
            break;
        case SolverType::FFT:
            forward_fft(ppinterval, n_pp, fs, seconds, bt,
                        thetai, basis_params, L, P, M, buf);
            break;
        case SolverType::RK3:
        case SolverType::RK4:
            forward_ode(ppinterval, n_pp, fs, seconds, bt, solver,
                        thetai, basis_params, L, P, M, buf);
            break;
    }
}

}}
