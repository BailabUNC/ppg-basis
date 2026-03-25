#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fastppg { namespace optim {

struct DEResult {
    std::vector<double> x;
    double fun;
    int nfev;
    int nit;
    bool converged;
};

struct DEConfig {
    int maxiter   = 60;
    int popsize   = 12;
    double tol    = 1e-2;
    int H         = 5;
    uint64_t seed = 42;
};

// Hypercube Sampling
inline void latin_hypercube(int NP, int D, const double* lb, const double* ub,
                            std::mt19937_64& rng, double* pop) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    for (int j = 0; j < D; ++j) {
        // create a permutation of [0, NP)
        std::vector<int> perm(NP);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int i = 0; i < NP; ++i) {
            double cell = (perm[i] + uni(rng)) / NP;
            pop[i * D + j] = lb[j] + cell * (ub[j] - lb[j]);
        }
    }
}

// Lehmer Mean
inline double weighted_lehmer_mean(const std::vector<double>& vals,
                                   const std::vector<double>& weights) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < vals.size(); ++i) {
        double w = weights[i];
        num += w * vals[i] * vals[i];
        den += w * vals[i];
    }
    return (den > 1e-30) ? num / den : 0.5;
}

// Arithmetic Mean
inline double weighted_mean(const std::vector<double>& vals,
                            const std::vector<double>& weights) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < vals.size(); ++i) {
        num += weights[i] * vals[i];
        den += weights[i];
    }
    return (den > 1e-30) ? num / den : 0.5;
}

// Differential Evolution
inline DEResult differential_evolution(
    std::function<double(const double* x, int thread_id)> cost_fn,
    int D,
    const double* lb, const double* ub,
    const double* x0,
    const DEConfig& cfg)
{
    int NP = cfg.popsize * D;
    if (NP < 4) NP = 4;

    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif

    std::vector<std::mt19937_64> rngs(num_threads);
    {
        std::mt19937_64 seed_rng(cfg.seed);
        for (int t = 0; t < num_threads; ++t)
            rngs[t].seed(seed_rng());
    }

    std::vector<double> pop(NP * D);
    latin_hypercube(NP, D, lb, ub, rngs[0], pop.data());

    // seed x0 as first individual
    if (x0 != nullptr) {
        for (int j = 0; j < D; ++j)
            pop[j] = x0[j];
    }

    std::vector<double> fitness(NP);
    int total_fev = 0;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NP; ++i) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        fitness[i] = cost_fn(pop.data() + i * D, tid);
    }
    total_fev += NP;

    // SHADE memory
    int H = cfg.H;
    std::vector<double> M_F(H, 0.5);
    std::vector<double> M_CR(H, 0.5);
    int mem_idx = 0;

    // trial vector storage
    std::vector<double> trial(D);

    // per-thread accumulators for successful F, CR, delta
    struct ThreadSuccess {
        std::vector<double> S_F, S_CR, S_delta;
        void clear() { S_F.clear(); S_CR.clear(); S_delta.clear(); }
    };
    std::vector<ThreadSuccess> thread_success(num_threads);

    DEResult result;
    result.nit = 0;
    result.converged = false;

    double prev_best = *std::min_element(fitness.begin(), fitness.end());
    int min_gens = std::max(5, cfg.maxiter / 10);  // at least 5 gens before convergence check

    for (int gen = 0; gen < cfg.maxiter; ++gen) {
        for (auto& ts : thread_success) ts.clear();

        struct IndivRandom {
            int r_mem;
            double F, CR;
            int r1, r2, r3;
            int j_rand;
            std::vector<double> u_cross;
        };
        std::vector<IndivRandom> ir(NP);
        {
            auto& rng = rngs[0];
            std::uniform_int_distribution<int> mem_dist(0, H - 1);
            std::cauchy_distribution<double> cauchy(0.0, 0.1);
            std::normal_distribution<double> normal(0.0, 0.1);
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            std::uniform_int_distribution<int> pop_dist(0, NP - 1);
            std::uniform_int_distribution<int> dim_dist(0, D - 1);

            for (int i = 0; i < NP; ++i) {
                ir[i].r_mem = mem_dist(rng);

                double F;
                do {
                    F = M_F[ir[i].r_mem] + cauchy(rng);
                } while (F <= 0.0);
                ir[i].F = std::min(F, 1.0);

                double CR = M_CR[ir[i].r_mem] + normal(rng);
                ir[i].CR = std::max(0.0, std::min(CR, 1.0));

                do { ir[i].r1 = pop_dist(rng); } while (ir[i].r1 == i);
                do { ir[i].r2 = pop_dist(rng); } while (ir[i].r2 == i || ir[i].r2 == ir[i].r1);
                do { ir[i].r3 = pop_dist(rng); } while (ir[i].r3 == i || ir[i].r3 == ir[i].r1 || ir[i].r3 == ir[i].r2);

                ir[i].j_rand = dim_dist(rng);
                ir[i].u_cross.resize(D);
                for (int j = 0; j < D; ++j) ir[i].u_cross[j] = uni(rng);
            }
        }

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < NP; ++i) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            const auto& ri = ir[i];
            std::vector<double> trial_vec(D);

            for (int j = 0; j < D; ++j) {
                if (ri.u_cross[j] < ri.CR || j == ri.j_rand) {
                    trial_vec[j] = pop[ri.r1 * D + j]
                                 + ri.F * (pop[ri.r2 * D + j] - pop[ri.r3 * D + j]);
                } else {
                    trial_vec[j] = pop[i * D + j];
                }
                if (trial_vec[j] < lb[j]) {
                    trial_vec[j] = lb[j] + std::fmod(std::fabs(trial_vec[j] - lb[j]),
                                                      ub[j] - lb[j]);
                }
                if (trial_vec[j] > ub[j]) {
                    trial_vec[j] = ub[j] - std::fmod(std::fabs(trial_vec[j] - ub[j]),
                                                      ub[j] - lb[j]);
                }
                // final clamp for safety
                trial_vec[j] = std::max(lb[j], std::min(trial_vec[j], ub[j]));
            }

            double trial_fit = cost_fn(trial_vec.data(), tid);

            if (trial_fit <= fitness[i]) {
                auto& ts = thread_success[tid];
                ts.S_F.push_back(ri.F);
                ts.S_CR.push_back(ri.CR);
                ts.S_delta.push_back(fitness[i] - trial_fit);

                for (int j = 0; j < D; ++j)
                    pop[i * D + j] = trial_vec[j];
                fitness[i] = trial_fit;
            }
        }
        total_fev += NP;

        std::vector<double> all_SF, all_SCR, all_Sdelta;
        for (auto& ts : thread_success) {
            all_SF.insert(all_SF.end(), ts.S_F.begin(), ts.S_F.end());
            all_SCR.insert(all_SCR.end(), ts.S_CR.begin(), ts.S_CR.end());
            all_Sdelta.insert(all_Sdelta.end(), ts.S_delta.begin(), ts.S_delta.end());
        }

        if (!all_SF.empty()) {
            M_F[mem_idx]  = weighted_lehmer_mean(all_SF, all_Sdelta);
            M_CR[mem_idx] = weighted_mean(all_SCR, all_Sdelta);
            mem_idx = (mem_idx + 1) % H;
        }

        double best = *std::min_element(fitness.begin(), fitness.end());
        result.nit = gen + 1;

        if (gen >= min_gens) {
            double mean_fit = 0.0;
            for (int i = 0; i < NP; ++i) mean_fit += fitness[i];
            mean_fit /= NP;
            double var_fit = 0.0;
            for (int i = 0; i < NP; ++i) {
                double d = fitness[i] - mean_fit;
                var_fit += d * d;
            }
            double std_fit = std::sqrt(var_fit / NP);
            double rel_spread = std_fit / (std::fabs(mean_fit) + 1e-30);
            double rel_imp = (prev_best - best) / (std::fabs(prev_best) + 1e-30);

            if (rel_spread < cfg.tol && rel_imp < cfg.tol) {
                result.converged = true;
                break;
            }
        }
        prev_best = best;
    }

    int best_idx = (int)(std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
    result.x.assign(pop.data() + best_idx * D, pop.data() + (best_idx + 1) * D);
    result.fun = fitness[best_idx];
    result.nfev = total_fev;
    if (!result.converged) result.converged = false;
    return result;
}

}}
