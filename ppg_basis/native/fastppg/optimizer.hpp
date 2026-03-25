#pragma once

#include "forward_models.hpp"
#include "cost_metrics.hpp"
#include "differential_evolution.hpp"
#include "local_optimizer.hpp"
#include "scratch_buffer.hpp"

#include <vector>
#include <cmath>
#include <string>
#include <functional>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fastppg { namespace optim {

struct ExtractConfig {
    std::string basis_type;
    std::string solver;
    int L;
    int M;
    int fs;

    // cost
    std::vector<std::string> cost_metric_names;
    std::vector<double> cost_weights;
    double max_nrmse = 5.0;

    // DE
    int de_maxiter  = 60;
    int de_popsize  = 12;
    double de_tol   = 1e-2;
    uint64_t seed   = 42;

    // local optimizer
    int slsqp_maxiter  = 1000;
    double slsqp_ftol  = 1e-8;

    // phase 2
    bool block_update  = true;
    int coord_cycles   = 4;
    int block_maxiter  = 300;
    double block_ftol  = 1e-6;
};

struct ExtractResult {
    std::vector<double> theta;   // [L]
    std::vector<double> params;  // [L * P]
    double final_cost;
    int total_fev;
};

inline ExtractResult extract_ppg(
    const double* signal, int n_signal,
    const double* pp_interval, int n_pp,
    const double* thetai_init,       // [L]
    const double* params_init,       // [L * P]
    const ExtractConfig& cfg)
{
    using namespace fastppg::models;
    using namespace fastppg::cost;

    BasisType bt = parse_basis_type(cfg.basis_type);
    SolverType sv = parse_solver(cfg.solver);
    int P = params_per_basis(bt);
    int D = cfg.L + cfg.L * P;

    int n_metrics = (int)cfg.cost_metric_names.size();
    std::vector<MetricType> metrics(n_metrics);
    for (int i = 0; i < n_metrics; ++i)
        metrics[i] = parse_metric(cfg.cost_metric_names[i]);

    const double* weights = cfg.cost_weights.data();

    double seconds = (double)n_signal / cfg.fs;
    int n_samples = (int)std::ceil(seconds * cfg.fs);

    // Bounds
    std::vector<double> lb(D), ub(D);

    // theta bounds: [-π, π]
    for (int i = 0; i < cfg.L; ++i) {
        lb[i] = -kernels::PI;
        ub[i] =  kernels::PI;
    }

    // param bounds from ppg_constants
    const double param_bnds_gaussian[][2]       = {{0.0,1.0},{0.05,3.0}};
    const double param_bnds_gamma[][2]          = {{0.0,1.0},{1.0,6.0},{0.05,3.0}};
    const double param_bnds_skewed_gaussian[][2] = {{0.0,1.0},{0.05,3.0},{-10.0,10.0}};

    const double (*bnds)[2] = nullptr;
    if (bt == BasisType::GAUSSIAN)        bnds = param_bnds_gaussian;
    else if (bt == BasisType::GAMMA)      bnds = param_bnds_gamma;
    else                                  bnds = param_bnds_skewed_gaussian;

    for (int i = 0; i < cfg.L; ++i) {
        for (int p = 0; p < P; ++p) {
            lb[cfg.L + i * P + p] = bnds[p][0];
            ub[cfg.L + i * P + p] = bnds[p][1];
        }
    }

    int n_constraints = (cfg.L > 1) ? cfg.L - 1 : 0;
    std::vector<double> constraint_data;
    if (n_constraints > 0) {
        constraint_data.resize(n_constraints * (D + 1), 0.0);
        for (int c = 0; c < n_constraints; ++c) {
            double* row = constraint_data.data() + c * (D + 1);
            row[c]     =  1.0;
            row[c + 1] = -1.0;
            row[D]     =  0.0;
        }
    }

    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif
    std::vector<ScratchBuffer> bufs;
    bufs.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t)
        bufs.emplace_back(n_samples, cfg.M, cfg.L, P);

    auto cost_fn = [&](const double* x, int tid) -> double {
        const double* theta = x;
        const double* params = x + cfg.L;
        auto& buf = bufs[tid];

        forward_model(pp_interval, n_pp, (double)cfg.fs, seconds,
                      bt, sv, theta, params, cfg.L, P, cfg.M, buf);

        double obj = objective_function(buf.z.data(), signal, n_samples, cfg.fs,
                                  metrics.data(), weights, n_metrics,
                                  cfg.max_nrmse, buf);

        for (int i = 0; i < cfg.L - 1; ++i) {
            double violation = theta[i] - theta[i + 1];
            if (violation > 0.0) obj += 1e4 * violation * violation;
        }
        return obj;
    };

    std::vector<double> x0(D);
    std::copy(thetai_init, thetai_init + cfg.L, x0.data());
    std::copy(params_init, params_init + cfg.L * P, x0.data() + cfg.L);

    {
        std::vector<int> order(cfg.L);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&x0](int a, int b) { return x0[a] < x0[b]; });
        std::vector<double> x0_sorted(D);
        for (int i = 0; i < cfg.L; ++i) {
            x0_sorted[i] = x0[order[i]];
            std::copy(x0.data() + cfg.L + order[i] * P,
                      x0.data() + cfg.L + (order[i] + 1) * P,
                      x0_sorted.data() + cfg.L + i * P);
        }
        x0 = x0_sorted;
    }

    // Phase 1 Differential Evolution
    DEConfig de_cfg;
    de_cfg.maxiter = cfg.de_maxiter;
    de_cfg.popsize = cfg.de_popsize;
    de_cfg.tol     = cfg.de_tol;
    de_cfg.seed    = cfg.seed;

    DEResult de_res = differential_evolution(cost_fn, D,
                                             lb.data(), ub.data(),
                                             x0.data(), de_cfg);

    {
        std::vector<int> order(cfg.L);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&de_res](int a, int b) { return de_res.x[a] < de_res.x[b]; });
        std::vector<double> sorted(D);
        for (int i = 0; i < cfg.L; ++i) {
            sorted[i] = de_res.x[order[i]];
            std::copy(de_res.x.data() + cfg.L + order[i] * P,
                      de_res.x.data() + cfg.L + (order[i] + 1) * P,
                      sorted.data() + cfg.L + i * P);
        }
        de_res.x = sorted;
    }

    int total_fev = de_res.nfev;

    auto cost_fn_single = [&](const double* x) -> double {
        return cost_fn(x, 0);
    };

    LocalConfig local_cfg;
    local_cfg.maxiter = cfg.slsqp_maxiter;
    local_cfg.ftol    = cfg.slsqp_ftol;

    LocalResult local_res = local_minimize(
        cost_fn_single, de_res.x.data(), D,
        lb.data(), ub.data(),
        n_constraints > 0 ? constraint_data.data() : nullptr,
        n_constraints, local_cfg);

    total_fev += local_res.nfev;

    // Phase 2 Block Coordinate Descent
    std::vector<double> best_x = local_res.x;

    if (cfg.block_update) {
        int block_D = 1 + P;
        std::vector<std::vector<double>> block_lb(cfg.L, std::vector<double>(block_D));
        std::vector<std::vector<double>> block_ub(cfg.L, std::vector<double>(block_D));
        for (int i = 0; i < cfg.L; ++i) {
            block_lb[i][0] = lb[i];
            block_ub[i][0] = ub[i];
            for (int p = 0; p < P; ++p) {
                block_lb[i][1 + p] = lb[cfg.L + i * P + p];
                block_ub[i][1 + p] = ub[cfg.L + i * P + p];
            }
        }

        // working copy of full state
        std::vector<double> working(best_x);

        for (int cycle = 0; cycle < cfg.coord_cycles; ++cycle) {
            for (int i = 0; i < cfg.L; ++i) {
                std::vector<double> xi0(block_D);
                xi0[0] = working[i];
                for (int p = 0; p < P; ++p)
                    xi0[1 + p] = working[cfg.L + i * P + p];

                auto block_cost = [&](const double* xi) -> double {
                    double old_theta = working[i];
                    std::vector<double> old_params(P);
                    for (int p = 0; p < P; ++p)
                        old_params[p] = working[cfg.L + i * P + p];

                    working[i] = xi[0];
                    for (int p = 0; p < P; ++p)
                        working[cfg.L + i * P + p] = xi[1 + p];

                    double val = cost_fn(working.data(), 0);

                    working[i] = old_theta;
                    for (int p = 0; p < P; ++p)
                        working[cfg.L + i * P + p] = old_params[p];

                    return val;
                };

                LocalConfig block_cfg;
                block_cfg.maxiter = cfg.block_maxiter;
                block_cfg.ftol    = cfg.block_ftol;

                LocalResult block_res = local_minimize(
                    block_cost, xi0.data(), block_D,
                    block_lb[i].data(), block_ub[i].data(),
                    nullptr, 0,
                    block_cfg);

                total_fev += block_res.nfev;

                working[i] = block_res.x[0];
                for (int p = 0; p < P; ++p)
                    working[cfg.L + i * P + p] = block_res.x[1 + p];
            }
        }
        best_x = working;
    }

    ExtractResult result;
    result.theta.assign(best_x.data(), best_x.data() + cfg.L);
    result.params.assign(best_x.data() + cfg.L, best_x.data() + D);
    result.final_cost = cost_fn(best_x.data(), 0);
    total_fev += 1;
    result.total_fev = total_fev;
    return result;
}

}}
