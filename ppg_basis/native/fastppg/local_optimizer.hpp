#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <limits>

namespace fastppg { namespace optim {

struct LocalResult {
    std::vector<double> x;
    double fun;
    int nfev;
    bool success;
};

struct LocalConfig {
    int maxiter    = 1000;
    double ftol    = 1e-8;
    double gtol    = 1e-6;
    double penalty = 1e4;   // for linear constraint violations
};

inline void fd_gradient(std::function<double(const double*)> f,
                        const double* x, int D,
                        const double* lb, const double* ub,
                        double* grad, int& nfev) {
    static constexpr double EPS = 1.4901161193847656e-08;
    std::vector<double> x_pert(x, x + D);
    for (int j = 0; j < D; ++j) {
        double h = EPS * std::max(1.0, std::fabs(x[j]));
        double x_fwd = std::min(x[j] + h, ub[j]);
        double x_bwd = std::max(x[j] - h, lb[j]);
        double actual_h = x_fwd - x_bwd;
        if (actual_h < 1e-30) {
            grad[j] = 0.0;
            continue;
        }
        x_pert[j] = x_fwd;
        double f_fwd = f(x_pert.data());
        x_pert[j] = x_bwd;
        double f_bwd = f(x_pert.data());
        x_pert[j] = x[j];
        grad[j] = (f_fwd - f_bwd) / actual_h;
        nfev += 2;
    }
}

inline void project_bounds(double* x, int D, const double* lb, const double* ub) {
    for (int j = 0; j < D; ++j)
        x[j] = std::max(lb[j], std::min(x[j], ub[j]));
}

inline double constraint_penalty(const double* x, int D,
                                 const double* constraint_data,
                                 int n_constraints) {
    double pen = 0.0;
    for (int c = 0; c < n_constraints; ++c) {
        const double* row = constraint_data + c * (D + 1);
        double ax = 0.0;
        for (int j = 0; j < D; ++j) ax += row[j] * x[j];
        double violation = ax - row[D]; // A·x - b ≤ 0 required
        if (violation > 0.0) pen += violation * violation;
    }
    return pen;
}

inline LocalResult local_minimize(
    std::function<double(const double*)> raw_objective,
    const double* x0, int D,
    const double* lb, const double* ub,
    const double* constraint_data,
    int n_constraints,
    const LocalConfig& cfg)
{
    auto augmented = [&](const double* x) -> double {
        double val = raw_objective(x);
        if (n_constraints > 0 && constraint_data != nullptr)
            val += cfg.penalty * constraint_penalty(x, D, constraint_data, n_constraints);
        return val;
    };

    std::vector<double> x(x0, x0 + D);
    project_bounds(x.data(), D, lb, ub);
    double fx = augmented(x.data());
    int nfev = 1;

    std::vector<double> grad(D);
    std::vector<double> x_prev(D);
    std::vector<double> grad_prev(D);
    std::vector<double> direction(D);

    constexpr int LBFGS_M = 7;
    std::vector<std::vector<double>> s_hist, y_hist;
    std::vector<double> rho_hist;

    fd_gradient(augmented, x.data(), D, lb, ub, grad.data(), nfev);

    for (int iter = 0; iter < cfg.maxiter; ++iter) {
        direction = grad;
        int m_used = (int)s_hist.size();

        std::vector<double> alpha_lbfgs(m_used);
        // first loop (backward)
        for (int i = m_used - 1; i >= 0; --i) {
            double dot_sy = 0.0;
            for (int j = 0; j < D; ++j) dot_sy += s_hist[i][j] * direction[j];
            alpha_lbfgs[i] = rho_hist[i] * dot_sy;
            for (int j = 0; j < D; ++j) direction[j] -= alpha_lbfgs[i] * y_hist[i][j];
        }
        if (m_used > 0) {
            double sy = 0.0, yy = 0.0;
            for (int j = 0; j < D; ++j) {
                sy += s_hist[m_used-1][j] * y_hist[m_used-1][j];
                yy += y_hist[m_used-1][j] * y_hist[m_used-1][j];
            }
            double gamma = (yy > 1e-30) ? sy / yy : 1.0;
            for (int j = 0; j < D; ++j) direction[j] *= gamma;
        }
        // second loop (forward)
        for (int i = 0; i < m_used; ++i) {
            double dot_yd = 0.0;
            for (int j = 0; j < D; ++j) dot_yd += y_hist[i][j] * direction[j];
            double beta = rho_hist[i] * dot_yd;
            for (int j = 0; j < D; ++j) direction[j] += (alpha_lbfgs[i] - beta) * s_hist[i][j];
        }
        // negate for descent
        for (int j = 0; j < D; ++j) direction[j] = -direction[j];

        double step = 1.0;
        double dir_grad = 0.0;
        for (int j = 0; j < D; ++j) dir_grad += grad[j] * direction[j];

        // if direction is not descent, fall back to steepest descent
        if (dir_grad >= 0.0) {
            for (int j = 0; j < D; ++j) direction[j] = -grad[j];
            dir_grad = 0.0;
            for (int j = 0; j < D; ++j) dir_grad += grad[j] * direction[j];
        }

        std::vector<double> x_new(D);
        double fx_new;
        bool step_found = false;
        for (int ls = 0; ls < 30; ++ls) {
            for (int j = 0; j < D; ++j) x_new[j] = x[j] + step * direction[j];
            project_bounds(x_new.data(), D, lb, ub);
            fx_new = augmented(x_new.data());
            nfev++;
            if (fx_new <= fx + 1e-4 * step * dir_grad) {
                step_found = true;
                break;
            }
            step *= 0.5;
        }
        if (!step_found) {
            // try steepest descent with tiny step
            step = 1e-6;
            for (int j = 0; j < D; ++j) x_new[j] = x[j] - step * grad[j];
            project_bounds(x_new.data(), D, lb, ub);
            fx_new = augmented(x_new.data());
            nfev++;
        }

        // Convergence Check
        double fx_change = std::fabs(fx - fx_new);
        if (fx_change < cfg.ftol * (std::fabs(fx) + 1e-30)) {
            x = x_new;
            fx = fx_new;
            break;
        }

        x_prev = x;
        grad_prev = grad;
        x = x_new;
        fx = fx_new;

        fd_gradient(augmented, x.data(), D, lb, ub, grad.data(), nfev);

        std::vector<double> s(D), y(D);
        double sy = 0.0;
        for (int j = 0; j < D; ++j) {
            s[j] = x[j] - x_prev[j];
            y[j] = grad[j] - grad_prev[j];
            sy += s[j] * y[j];
        }
        if (sy > 1e-30) {
            if ((int)s_hist.size() >= LBFGS_M) {
                s_hist.erase(s_hist.begin());
                y_hist.erase(y_hist.begin());
                rho_hist.erase(rho_hist.begin());
            }
            s_hist.push_back(std::move(s));
            y_hist.push_back(std::move(y));
            rho_hist.push_back(1.0 / sy);
        }

        double gnorm = 0.0;
        for (int j = 0; j < D; ++j) gnorm += grad[j] * grad[j];
        if (std::sqrt(gnorm) < cfg.gtol) break;
    }

    LocalResult res;
    res.x = x;
    res.fun = raw_objective(x.data());
    res.nfev = nfev;
    res.success = true;
    return res;
}

}}
