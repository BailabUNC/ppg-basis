#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <cstring>
#include <chrono>

#include "kernels.hpp"
#include "fft_utils.hpp"
#include "scratch_buffer.hpp"
#include "forward_models.hpp"
#include "cost_metrics.hpp"
#include "differential_evolution.hpp"
#include "local_optimizer.hpp"
#include "optimizer.hpp"

using namespace fastppg;

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "  FAIL: " << msg << " [" << __FILE__ << ":" << __LINE__ << "]\n"; \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define SECTION(name) std::cout << "\n── " << name << " ──\n"

// Kernel Tests
void test_wrap_pi() {
    SECTION("kernels::wrap_pi");
    CHECK(std::fabs(kernels::wrap_pi(0.0)) < 1e-15, "wrap_pi(0) == 0");
    CHECK(std::fabs(kernels::wrap_pi(kernels::PI) - kernels::PI) < 1e-10
       || std::fabs(kernels::wrap_pi(kernels::PI) + kernels::PI) < 1e-10,
          "wrap_pi(π) is ±π");
    CHECK(std::fabs(kernels::wrap_pi(3.0 * kernels::PI) - kernels::PI) < 1e-10
       || std::fabs(kernels::wrap_pi(3.0 * kernels::PI) + kernels::PI) < 1e-10,
          "wrap_pi(3π) wraps correctly");
    double v = kernels::wrap_pi(-4.0);
    CHECK(v >= -kernels::PI && v <= kernels::PI, "wrap_pi(-4) in [-π,π]");
}

void test_corrcoef() {
    SECTION("kernels::corrcoef");
    double x[5] = {1, 2, 3, 4, 5};
    double y[5] = {2, 4, 6, 8, 10};
    CHECK(std::fabs(kernels::corrcoef(x, y, 5) - 1.0) < 1e-6,
          "perfect correlation ≈ 1.0");
    double y2[5] = {10, 8, 6, 4, 2};
    CHECK(std::fabs(kernels::corrcoef(x, y2, 5) + 1.0) < 1e-6,
          "perfect anticorrelation ≈ -1.0");
}

void test_gamma_pdf_precomp() {
    SECTION("kernels::gamma_pdf_precomp");
    double alpha = 3.0, scale = 0.5;
    double alpha_m1 = alpha - 1.0;
    double inv_scale = 1.0 / scale;
    double log_norm = std::lgamma(alpha) + alpha * std::log(scale);

    double x = 1.0;
    double val = kernels::gamma_pdf_precomp(x, alpha_m1, inv_scale, log_norm);
    double ref = kernels::gamma_pdf(x, alpha, scale);
    CHECK(std::fabs(val - ref) < 1e-14, "precomp matches original gamma_pdf");
    CHECK(kernels::gamma_pdf_precomp(-0.1, alpha_m1, inv_scale, log_norm) == 0.0,
          "gamma_pdf(x<=0) = 0");
}

void test_gradient_1d() {
    SECTION("kernels::gradient_1d");
    int n = 100;
    std::vector<double> arr(n), grad(n);
    double dx = 0.1;
    for (int i = 0; i < n; ++i) arr[i] = 2.0 * i * dx + 1.0;  // y = 2x + 1
    kernels::gradient_1d(arr.data(), n, dx, grad.data());
    for (int i = 1; i < n - 1; ++i)
        CHECK(std::fabs(grad[i] - 2.0) < 1e-10, "linear gradient = 2.0");
}

void test_detrend_linear() {
    SECTION("kernels::detrend_linear");
    int n = 200;
    std::vector<double> arr(n);
    for (int i = 0; i < n; ++i)
        arr[i] = 5.0 + 0.1 * i + 0.5 * std::sin(2.0 * kernels::PI * i / 50.0);
    kernels::detrend_linear(arr.data(), n);
    double mean = 0;
    for (int i = 0; i < n; ++i) mean += arr[i];
    mean /= n;
    CHECK(std::fabs(mean) < 1e-10, "detrended mean ≈ 0");
}

void test_postprocess_z() {
    SECTION("kernels::postprocess_z");
    int n = 100;
    std::vector<double> z(n);
    for (int i = 0; i < n; ++i) z[i] = std::sin(2.0 * kernels::PI * i / 30.0) + 0.01 * i;
    z[5] = std::nan("");
    z[10] = std::numeric_limits<double>::infinity();
    kernels::postprocess_z(z.data(), n);
    // output should be in [0, 1]
    double mn = *std::min_element(z.begin(), z.end());
    double mx = *std::max_element(z.begin(), z.end());
    CHECK(mn >= -1e-10, "postprocess min >= 0");
    CHECK(mx <= 1.0 + 1e-10, "postprocess max <= 1");
    // no NaN/Inf
    for (int i = 0; i < n; ++i)
        CHECK(std::isfinite(z[i]), "postprocess output is finite");
}

void test_fft() {
    SECTION("fft_utils");
    int M = 64;
    std::vector<double> signal(M);
    for (int i = 0; i < M; ++i)
        signal[i] = std::cos(2.0 * kernels::PI * 3.0 * i / M); // freq = 3

    std::vector<fft::cdouble> spectrum(M);
    fft::fft_forward(signal.data(), M, spectrum.data());

    double peak3 = std::abs(spectrum[3]);
    double peak_m3 = std::abs(spectrum[M - 3]);
    double noise = 0;
    for (int k = 0; k < M; ++k) {
        if (k != 3 && k != M - 3) noise += std::abs(spectrum[k]);
    }
    CHECK(peak3 > 10.0, "FFT peak at k=3");
    CHECK(peak_m3 > 10.0, "FFT peak at k=M-3");
    CHECK(noise < 1e-10, "FFT noise floor ≈ 0");

    std::vector<double> recovered(M);
    fft::fft_inverse_real(spectrum.data(), M, recovered.data());
    for (int i = 0; i < M; ++i)
        CHECK(std::fabs(recovered[i] - signal[i]) < 1e-10, "FFT roundtrip");

    int M2 = 60;
    std::vector<double> sig2(M2);
    for (int i = 0; i < M2; ++i)
        sig2[i] = std::cos(2.0 * kernels::PI * 5.0 * i / M2);
    std::vector<fft::cdouble> spec2(M2);
    fft::fft_forward(sig2.data(), M2, spec2.data());
    std::vector<double> rec2(M2);
    fft::fft_inverse_real(spec2.data(), M2, rec2.data());
    for (int i = 0; i < M2; ++i)
        CHECK(std::fabs(rec2[i] - sig2[i]) < 1e-8, "Bluestein roundtrip");
}

void test_forward_models() {
    SECTION("forward_models (all solver × basis combos)");

    int L = 3, M = 512, fs = 60;
    double duration = 2.0;
    int n_samples = (int)std::ceil(duration * fs);

    double thetai[3] = {-1.5, 0.0, 1.5};
    double params_g[6]  = {0.8,0.3, 0.5,0.5, 0.3,0.2};          // gaussian: L×2
    double params_ga[9] = {0.5,2.0,0.5, 0.3,3.0,0.3, 0.2,4.0,0.2}; // gamma: L×3
    double params_sg[9] = {0.5,0.3,1.0, 0.3,0.5,-2.0, 0.2,0.4,0.5}; // skewed-gaussian: L×3
    double pp[3] = {1.0, 1.0, 1.0};

    struct TestCase {
        const char* bt_name;
        models::BasisType bt;
        double* params;
        int P;
    };
    TestCase cases[] = {
        {"gaussian",        models::BasisType::GAUSSIAN,        params_g,  2},
        {"gamma",           models::BasisType::GAMMA,           params_ga, 3},
        {"skewed-gaussian", models::BasisType::SKEWED_GAUSSIAN, params_sg, 3},
    };

    models::SolverType solvers[] = {
        models::SolverType::BASIS,
        models::SolverType::TEMPLATE,
        models::SolverType::FFT,
        models::SolverType::RK3,
        models::SolverType::RK4,
    };
    const char* solver_names[] = {"basis", "template", "fft", "rk3", "rk4"};

    for (auto& tc : cases) {
        for (int si = 0; si < 5; ++si) {
            ScratchBuffer buf(n_samples, M, L, tc.P);
            models::forward_model(pp, 3, (double)fs, duration,
                                  tc.bt, solvers[si],
                                  thetai, tc.params, L, tc.P, M, buf);

            bool all_finite = true;
            double mn = 1e30, mx = -1e30;
            for (int i = 0; i < n_samples; ++i) {
                if (!std::isfinite(buf.z[i])) all_finite = false;
                if (buf.z[i] < mn) mn = buf.z[i];
                if (buf.z[i] > mx) mx = buf.z[i];
            }

            char msg[128];
            snprintf(msg, sizeof(msg), "%s/%s: all finite", solver_names[si], tc.bt_name);
            CHECK(all_finite, msg);
            snprintf(msg, sizeof(msg), "%s/%s: min >= 0", solver_names[si], tc.bt_name);
            CHECK(mn >= -1e-10, msg);
            snprintf(msg, sizeof(msg), "%s/%s: max <= 1", solver_names[si], tc.bt_name);
            CHECK(mx <= 1.0 + 1e-10, msg);
        }
    }
}

void test_cost_metrics() {
    SECTION("cost_metrics");

    int n = 120;
    std::vector<double> model(n), signal(n);
    for (int i = 0; i < n; ++i) {
        model[i] = 0.5 + 0.5 * std::sin(2.0 * kernels::PI * i / 30.0);
        signal[i] = model[i] + 0.05 * std::sin(2.0 * kernels::PI * i / 7.0);
    }
    auto norm = [](std::vector<double>& v) {
        double mn = *std::min_element(v.begin(), v.end());
        double mx = *std::max_element(v.begin(), v.end());
        for (auto& x : v) x = (x - mn) / (mx - mn + 1e-8);
    };
    norm(model); norm(signal);

    double mse_val = cost::mse(model.data(), signal.data(), n);
    CHECK(mse_val >= 0.0, "MSE >= 0");
    CHECK(mse_val < 0.1, "MSE reasonable for similar signals");

    CHECK(cost::mse(model.data(), model.data(), n) < 1e-15, "MSE(x,x) ≈ 0");

    double corr_val = cost::corr(model.data(), signal.data(), n);
    CHECK(corr_val >= 0.0, "1 - pearson >= 0 for similar signals");
    CHECK(corr_val < 0.1, "corr cost small for similar signals");

    ScratchBuffer buf(n, 1, 1, 1);
    double appg_val = cost::appg(model.data(), signal.data(), n, 60, buf);
    CHECK(std::isfinite(appg_val), "APPG is finite");
    CHECK(appg_val >= 0.0, "APPG >= 0");

    double sp_val = cost::spearman(model.data(), signal.data(), n, buf);
    CHECK(std::isfinite(sp_val), "Spearman is finite");
    CHECK(sp_val >= 0.0, "Spearman >= 0");

    double kt_val = cost::kendall(model.data(), signal.data(), n);
    CHECK(kt_val >= 0.0 && kt_val <= 1.0 + 1e-10, "Kendall in [0,1]");

    cost::MetricType mts[2] = {cost::MetricType::MSE, cost::MetricType::CORR};
    double wts[2] = {1.0, 1.0};
    double obj = cost::objective_function(model.data(), signal.data(), n, 60,
                                          mts, wts, 2, 5.0, buf);
    CHECK(std::isfinite(obj) && obj >= 0.0, "objective_function finite and >= 0");
}

void test_differential_evolution() {
    SECTION("differential_evolution (2D Rosenbrock)");

    auto rosenbrock = [](const double* x, int /*tid*/) -> double {
        double a = 1.0 - x[0];
        double b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    };

    double lb[2] = {-5.0, -5.0};
    double ub[2] = { 5.0,  5.0};
    double x0[2] = { 0.0,  0.0};

    optim::DEConfig cfg;
    cfg.maxiter = 300;
    cfg.popsize = 15;
    cfg.tol     = 1e-10;
    cfg.seed    = 42;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = optim::differential_evolution(rosenbrock, 2, lb, ub, x0, cfg);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  Rosenbrock result: x=(" << result.x[0] << ", " << result.x[1]
              << "), f=" << result.fun << ", nfev=" << result.nfev
              << ", time=" << ms << "ms\n";

    CHECK(std::fabs(result.x[0] - 1.0) < 0.01, "x ≈ 1.0");
    CHECK(std::fabs(result.x[1] - 1.0) < 0.01, "y ≈ 1.0");
    CHECK(result.fun < 1e-4, "f(x,y) ≈ 0");
}

void test_local_optimizer() {
    SECTION("local_optimizer (bounded Rosenbrock)");

    auto rosenbrock = [](const double* x) -> double {
        double a = 1.0 - x[0];
        double b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    };

    double x0[2] = {0.5, 0.5};
    double lb[2] = {-2.0, -2.0};
    double ub[2] = { 2.0,  2.0};

    optim::LocalConfig cfg;
    cfg.maxiter = 500;
    cfg.ftol = 1e-12;

    auto result = optim::local_minimize(rosenbrock, x0, 2, lb, ub,
                                         nullptr, 0, cfg);

    std::cout << "  Local result: x=(" << result.x[0] << ", " << result.x[1]
              << "), f=" << result.fun << ", nfev=" << result.nfev << "\n";

    CHECK(std::fabs(result.x[0] - 1.0) < 0.01, "local x ≈ 1.0");
    CHECK(std::fabs(result.x[1] - 1.0) < 0.01, "local y ≈ 1.0");
}

void test_full_pipeline() {
    SECTION("optimizer::extract_ppg (gaussian/basis round-trip)");

    int L = 3, fs = 60, M = 512;
    double duration = 2.0;
    int n_samples = (int)std::ceil(duration * fs);

    // ground truth parameters
    double true_theta[3] = {-1.5, 0.0, 1.5};
    double true_params[6] = {0.8, 0.3,  0.5, 0.5,  0.3, 0.2};
    double pp[3] = {1.0, 1.0, 1.0};
    int n_pp = 3;

    // generate ground truth signal
    ScratchBuffer gen_buf(n_samples, M, L, 2);
    models::forward_model(pp, n_pp, (double)fs, duration,
                          models::BasisType::GAUSSIAN,
                          models::SolverType::BASIS,
                          true_theta, true_params, L, 2, M, gen_buf);
    std::vector<double> signal(gen_buf.z.begin(), gen_buf.z.begin() + n_samples);

    // initial guess
    double init_theta[3] = {-2.0, -0.5, 2.0};
    double init_params[6] = {0.5, 0.5,  0.5, 0.5,  0.5, 0.5};

    // configure extraction
    optim::ExtractConfig cfg;
    cfg.basis_type = "gaussian";
    cfg.solver = "basis";
    cfg.L = L;
    cfg.M = M;
    cfg.fs = fs;
    cfg.cost_metric_names = {"mse", "corr"};
    cfg.cost_weights = {1.0, 1.0};
    cfg.de_maxiter = 60;
    cfg.de_popsize = 12;
    cfg.de_tol = 1e-3;
    cfg.slsqp_maxiter = 500;
    cfg.slsqp_ftol = 1e-8;
    cfg.block_update = true;
    cfg.coord_cycles = 4;
    cfg.block_maxiter = 200;
    cfg.block_ftol = 1e-6;
    cfg.seed = 42;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = optim::extract_ppg(signal.data(), n_samples,
                                      pp, n_pp,
                                      init_theta, init_params, cfg);
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    // reconstruct from extracted params
    ScratchBuffer recon_buf(n_samples, M, L, 2);
    models::forward_model(pp, n_pp, (double)fs, duration,
                          models::BasisType::GAUSSIAN,
                          models::SolverType::BASIS,
                          result.theta.data(), result.params.data(),
                          L, 2, M, recon_buf);

    double mse = 0.0;
    for (int i = 0; i < n_samples; ++i) {
        double d = signal[i] - recon_buf.z[i];
        mse += d * d;
    }
    mse /= n_samples;

    std::cout << "  Extraction time: " << sec << "s\n";
    std::cout << "  Final cost: " << result.final_cost << "\n";
    std::cout << "  Round-trip MSE: " << mse << "\n";
    std::cout << "  Total fev: " << result.total_fev << "\n";
    std::cout << "  Extracted theta: [" << result.theta[0] << ", "
              << result.theta[1] << ", " << result.theta[2] << "]\n";

    CHECK(mse < 0.05, "round-trip MSE < 0.05");
    CHECK(result.final_cost < 1.0, "final cost reasonable");
    CHECK(sec < 30.0, "extraction < 30s");

    // theta ordering
    bool ordered = true;
    for (int i = 0; i < L - 1; ++i) {
        if (result.theta[i] > result.theta[i + 1] + 0.05) ordered = false;
    }
    CHECK(ordered, "extracted thetas roughly ordered");
}

// ── Main ────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "  fastppg standalone test suite\n";

#ifdef _OPENMP
    std::cout << "  OpenMP threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "  OpenMP: not available\n";
#endif

    test_wrap_pi();
    test_corrcoef();
    test_gamma_pdf_precomp();
    test_gradient_1d();
    test_detrend_linear();
    test_postprocess_z();
    test_fft();
    test_forward_models();
    test_cost_metrics();
    test_differential_evolution();
    test_local_optimizer();
    test_full_pipeline();

    std::cout << "  Results: " << tests_passed << " passed, "
              << tests_failed << " failed\n";

    return tests_failed > 0 ? 1 : 0;
}
