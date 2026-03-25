#pragma once

#include "kernels.hpp"
#include "scratch_buffer.hpp"
#include <cmath>
#include <algorithm>
#include <string>
#include <stdexcept>

namespace fastppg { namespace cost {

using namespace fastppg::kernels;

// Metric type enum
enum class MetricType : uint8_t { MSE, CORR, APPG, SPEARMAN, KENDALL, GAMMA, SOMERS };

inline MetricType parse_metric(const std::string& name) {
    if (name == "mse")      return MetricType::MSE;
    if (name == "corr")     return MetricType::CORR;
    if (name == "appg")     return MetricType::APPG;
    if (name == "spearman") return MetricType::SPEARMAN;
    if (name == "kendall")  return MetricType::KENDALL;
    if (name == "gamma")    return MetricType::GAMMA;
    if (name == "somers")   return MetricType::SOMERS;
    throw std::runtime_error("Unknown cost metric: " + name);
}

// Raw Metrics
inline double mse(const double* model, const double* signal, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = model[i] - signal[i];
        acc += d * d;
    }
    return acc / n;
}

inline double corr(const double* model, const double* signal, int n) {
    return 1.0 - corrcoef(model, signal, n);
}

// NRMSE of 2nd derivative of PPG
inline double appg(const double* model, const double* signal, int n, int fs,
                   ScratchBuffer& buf) {
    double dt = 1.0 / fs;
    // smooth signal → d/dt → d²/dt²
    gaussian_filter1d(signal, n, 2.0, buf.smooth1.data());
    gradient_1d(buf.smooth1.data(), n, dt, buf.grad1.data());
    gradient_1d(buf.grad1.data(), n, dt, buf.grad2.data());
    // smooth model → d/dt → d²/dt²
    gaussian_filter1d(model, n, 2.0, buf.smooth2.data());
    gradient_1d(buf.smooth2.data(), n, dt, buf.grad3.data());
    gradient_1d(buf.grad3.data(), n, dt, buf.grad4.data());

    double mean_d2sig = 0.0;
    for (int i = 0; i < n; ++i) mean_d2sig += buf.grad2[i];
    mean_d2sig /= n;

    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = buf.grad4[i] - buf.grad2[i];
        num += diff * diff;
        double dev = buf.grad2[i] - mean_d2sig;
        den += dev * dev;
    }
    if (den < 1e-12) return 0.0;
    return std::sqrt(num / den);
}

inline double spearman(const double* model, const double* signal, int n,
                       ScratchBuffer& buf) {
    rank_array(model, n, buf.ranks1.data());
    rank_array(signal, n, buf.ranks2.data());
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = buf.ranks1[i] - buf.ranks2[i];
        acc += d * d;
    }
    return acc;
}

inline double kendall(const double* model, const double* signal, int n) {
    int concordant = 0, discordant = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double md = model[i] - model[j];
            double sd = signal[i] - signal[j];
            if (md * sd > 0.0) concordant++;
            else if (md * sd < 0.0) discordant++;
        }
    }
    return (double)discordant / (concordant + discordant + 1e-8);
}

inline double gamma_metric(const double* model, const double* signal, int n) {
    int concordant = 0, discordant = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double md = model[i] - model[j];
            double sd = signal[i] - signal[j];
            if (md != 0.0 && sd != 0.0) {
                if (md * sd > 0.0) concordant++;
                else if (md * sd < 0.0) discordant++;
            }
        }
    }
    return (double)discordant / (concordant + discordant + 1e-8);
}

inline double somers_d(const double* model, const double* signal, int n) {
    int concordant = 0, discordant = 0, ties_y = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double md = model[i] - model[j];
            double sd = signal[i] - signal[j];
            if (sd == 0.0) ties_y++;
            else if (md * sd > 0.0) concordant++;
            else if (md * sd < 0.0) discordant++;
        }
    }
    return (double)(discordant + ties_y) / (concordant + discordant + ties_y + 1e-8);
}

// Normalizers
inline double normalize_identity(double raw) {
    return std::max(0.0, std::min(raw, 1.0));
}

inline double normalize_corr(double raw) {
    return std::max(0.0, std::min(raw / 2.0, 1.0));
}

inline double normalize_appg(double raw, double max_nrmse = 5.0) {
    if (max_nrmse > 0.0)
        return std::max(0.0, std::min(raw / max_nrmse, 1.0));
    return 1.0 / (1.0 + std::exp(-raw));  // logistic fallback
}

inline double normalize_spearman(double raw, int n) {
    double max_s = (n >= 2) ? n * ((double)n * n - 1.0) / 3.0 : 1.0;
    return std::max(0.0, std::min(raw / max_s, 1.0));
}

// Objective Function
inline double compute_metric_normalized(MetricType mt,
                                        const double* model,
                                        const double* signal, int n,
                                        int fs, double max_nrmse,
                                        ScratchBuffer& buf) {
    double raw;
    switch (mt) {
        case MetricType::MSE:
            raw = mse(model, signal, n);
            return normalize_identity(raw);
        case MetricType::CORR:
            raw = corr(model, signal, n);
            return normalize_corr(raw);
        case MetricType::APPG:
            raw = appg(model, signal, n, fs, buf);
            return normalize_appg(raw, max_nrmse);
        case MetricType::SPEARMAN:
            raw = spearman(model, signal, n, buf);
            return normalize_spearman(raw, n);
        case MetricType::KENDALL:
            raw = kendall(model, signal, n);
            return normalize_identity(raw);
        case MetricType::GAMMA:
            raw = gamma_metric(model, signal, n);
            return normalize_identity(raw);
        case MetricType::SOMERS:
            raw = somers_d(model, signal, n);
            return normalize_identity(raw);
    }
    return 0.0;
}

inline double objective_function(const double* model, const double* signal,
                                 int n, int fs,
                                 const MetricType* metrics,
                                 const double* weights, int n_metrics,
                                 double max_nrmse,
                                 ScratchBuffer& buf) {
    double total = 0.0;
    for (int i = 0; i < n_metrics; ++i) {
        double val = compute_metric_normalized(metrics[i], model, signal,
                                               n, fs, max_nrmse, buf);
        total += val * weights[i];
    }
    return total;
}

}}
