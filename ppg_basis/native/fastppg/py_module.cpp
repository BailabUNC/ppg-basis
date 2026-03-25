#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "kernels.hpp"
#include "fft_utils.hpp"
#include "forward_models.hpp"
#include "cost_metrics.hpp"
#include "differential_evolution.hpp"
#include "local_optimizer.hpp"
#include "optimizer.hpp"
#include "scratch_buffer.hpp"

namespace py = pybind11;
using arr_d = py::array_t<double, py::array::c_style | py::array::forcecast>;

PYBIND11_MODULE(fastppg, m) {
    m.doc() = "Native C++ acceleration for ppg_basis";

    m.def("corrcoef", [](arr_d x, arr_d y) {
        auto bx = x.request(); auto by = y.request();
        if (bx.ndim != 1 || by.ndim != 1 || bx.shape[0] != by.shape[0])
            throw std::runtime_error("corrcoef: arrays must be 1D and same length");
        return fastppg::kernels::corrcoef((double*)bx.ptr, (double*)by.ptr,
                                           (int)bx.shape[0]);
    }, "Pearson's r between two 1D arrays");

    m.def("gaussian_filter1d", [](arr_d arr, double sigma) -> arr_d {
        auto buf = arr.request();
        int n = (int)buf.shape[0];
        arr_d out({n});
        fastppg::kernels::gaussian_filter1d((double*)buf.ptr, n, sigma,
                                             (double*)out.mutable_data());
        return out;
    });

    m.def("gradient_1d", [](arr_d arr, double dx) -> arr_d {
        auto buf = arr.request();
        int n = (int)buf.shape[0];
        arr_d out({n});
        fastppg::kernels::gradient_1d((double*)buf.ptr, n, dx,
                                       (double*)out.mutable_data());
        return out;
    });

    m.def("detrend_linear", [](arr_d arr) -> arr_d {
        auto buf = arr.request();
        int n = (int)buf.shape[0];
        arr_d out({n});
        std::memcpy(out.mutable_data(), buf.ptr, n * sizeof(double));
        fastppg::kernels::detrend_linear((double*)out.mutable_data(), n);
        return out;
    });

    m.def("postprocess_z", [](arr_d arr) -> arr_d {
        auto buf = arr.request();
        int n = (int)buf.shape[0];
        arr_d out({n});
        std::memcpy(out.mutable_data(), buf.ptr, n * sizeof(double));
        fastppg::kernels::postprocess_z((double*)out.mutable_data(), n);
        return out;
    });

    m.def("gamma_pdf_precomp", [](double x, double alpha_m1,
                                   double inv_scale, double log_norm) {
        return fastppg::kernels::gamma_pdf_precomp(x, alpha_m1, inv_scale, log_norm);
    });

    // Forward Models
    m.def("forward_model", [](arr_d pp, int fs, double seconds,
                               const std::string& basis_type,
                               arr_d thetai, arr_d params,
                               const std::string& solver, int M) -> arr_d {
        auto pp_buf = pp.request();
        auto th_buf = thetai.request();
        auto pa_buf = params.request();

        int n_pp = (int)pp_buf.shape[0];
        int L = (int)th_buf.shape[0];

        auto bt = fastppg::models::parse_basis_type(basis_type);
        auto sv = fastppg::models::parse_solver(solver);
        int P = fastppg::models::params_per_basis(bt);
        int n_samples = (int)std::ceil(seconds * fs);

        arr_d out({n_samples});
        double* out_ptr = (double*)out.mutable_data();

        const double* pp_ptr = (double*)pp_buf.ptr;
        const double* th_ptr = (double*)th_buf.ptr;
        const double* pa_ptr = (double*)pa_buf.ptr;

        {
            py::gil_scoped_release release;
            fastppg::ScratchBuffer buf(n_samples, M, L, P);

            fastppg::models::forward_model(
                pp_ptr, n_pp, (double)fs, seconds,
                bt, sv, th_ptr, pa_ptr,
                L, P, M, buf);

            std::memcpy(out_ptr, buf.z.data(), n_samples * sizeof(double));
        }

        return out;
    }, "Run a forward model and return the PPG waveform",
    py::arg("pp"), py::arg("fs"), py::arg("seconds"),
    py::arg("basis_type"), py::arg("thetai"), py::arg("params"),
    py::arg("solver"), py::arg("M"));

    // Cost Metrics
    m.def("cost_mse", [](arr_d model, arr_d signal) {
        auto mb = model.request(); auto sb = signal.request();
        return fastppg::cost::mse((double*)mb.ptr, (double*)sb.ptr, (int)mb.shape[0]);
    });

    m.def("cost_corr", [](arr_d model, arr_d signal) {
        auto mb = model.request(); auto sb = signal.request();
        return fastppg::cost::corr((double*)mb.ptr, (double*)sb.ptr, (int)mb.shape[0]);
    });

    m.def("cost_appg", [](arr_d model, arr_d signal, int fs) {
        auto mb = model.request(); auto sb = signal.request();
        int n = (int)mb.shape[0];
        fastppg::ScratchBuffer buf(n, 1, 1, 1);
        return fastppg::cost::appg((double*)mb.ptr, (double*)sb.ptr, n, fs, buf);
    });

    m.def("cost_spearman", [](arr_d model, arr_d signal) {
        auto mb = model.request(); auto sb = signal.request();
        int n = (int)mb.shape[0];
        fastppg::ScratchBuffer buf(n, 1, 1, 1);
        return fastppg::cost::spearman((double*)mb.ptr, (double*)sb.ptr, n, buf);
    });

    m.def("cost_kendall", [](arr_d model, arr_d signal) {
        auto mb = model.request(); auto sb = signal.request();
        return fastppg::cost::kendall((double*)mb.ptr, (double*)sb.ptr, (int)mb.shape[0]);
    });

    m.def("objective_function", [](arr_d model, arr_d signal,
                                    py::list metric_names, py::list weights_list,
                                    int fs, double max_nrmse) {
        auto mb = model.request(); auto sb = signal.request();
        int n = (int)mb.shape[0];

        int n_m = (int)py::len(metric_names);
        std::vector<fastppg::cost::MetricType> mts(n_m);
        std::vector<double> wts(n_m);
        for (int i = 0; i < n_m; ++i) {
            mts[i] = fastppg::cost::parse_metric(metric_names[i].cast<std::string>());
            wts[i] = weights_list[i].cast<double>();
        }

        fastppg::ScratchBuffer buf(n, 1, 1, 1);
        return fastppg::cost::objective_function(
            (double*)mb.ptr, (double*)sb.ptr, n, fs,
            mts.data(), wts.data(), n_m, max_nrmse, buf);
    }, py::arg("model"), py::arg("signal"), py::arg("metric_names"),
       py::arg("weights"), py::arg("fs"), py::arg("max_nrmse") = 5.0);

    // Extraction Pipeline
    m.def("extract_ppg_native", [](
        arr_d signal, arr_d pp_interval,
        int fs, const std::string& basis_type, const std::string& solver,
        int L, int M,
        py::list cost_metrics, py::list cost_weights_list,
        arr_d thetai_init, arr_d params_init,
        int maxiter_de, int popsize, double tol_de,
        int maxiter_slsqp, double ftol_slsqp,
        bool block_update, int coord_cycles,
        uint64_t seed
    ) -> py::tuple {
        auto sig_buf = signal.request();
        auto pp_buf  = pp_interval.request();
        auto th_buf  = thetai_init.request();
        auto pa_buf  = params_init.request();

        int n_signal = (int)sig_buf.shape[0];
        int n_pp     = (int)pp_buf.shape[0];

        fastppg::optim::ExtractConfig cfg;
        cfg.basis_type = basis_type;
        cfg.solver     = solver;
        cfg.L          = L;
        cfg.M          = M;
        cfg.fs         = fs;
        cfg.de_maxiter = maxiter_de;
        cfg.de_popsize = popsize;
        cfg.de_tol     = tol_de;
        cfg.slsqp_maxiter = maxiter_slsqp;
        cfg.slsqp_ftol    = ftol_slsqp;
        cfg.block_update  = block_update;
        cfg.coord_cycles  = coord_cycles;
        cfg.seed          = seed;

        int n_m = (int)py::len(cost_metrics);
        for (int i = 0; i < n_m; ++i) {
            cfg.cost_metric_names.push_back(cost_metrics[i].cast<std::string>());
            cfg.cost_weights.push_back(cost_weights_list[i].cast<double>());
        }

        const double* sig_ptr = (double*)sig_buf.ptr;
        const double* pp_ptr  = (double*)pp_buf.ptr;
        const double* th_ptr  = (double*)th_buf.ptr;
        const double* pa_ptr  = (double*)pa_buf.ptr;

        fastppg::optim::ExtractResult result;

        {
            py::gil_scoped_release release;
            result = fastppg::optim::extract_ppg(
                sig_ptr, n_signal, pp_ptr, n_pp,
                th_ptr, pa_ptr, cfg);
        }

        // build output arrays
        auto bt = fastppg::models::parse_basis_type(basis_type);
        int P = fastppg::models::params_per_basis(bt);

        arr_d theta_out({L});
        arr_d params_out({L, P});
        std::memcpy(theta_out.mutable_data(), result.theta.data(),
                     L * sizeof(double));
        std::memcpy(params_out.mutable_data(), result.params.data(),
                     L * P * sizeof(double));

        return py::make_tuple(theta_out, params_out, result.final_cost,
                              result.total_fev);
    },
    py::arg("signal"), py::arg("pp_interval"),
    py::arg("fs"), py::arg("basis_type"), py::arg("solver"),
    py::arg("L"), py::arg("M"),
    py::arg("cost_metrics"), py::arg("cost_weights"),
    py::arg("thetai_init"), py::arg("params_init"),
    py::arg("maxiter_de") = 60, py::arg("popsize") = 12,
    py::arg("tol_de") = 1e-2,
    py::arg("maxiter_slsqp") = 1000, py::arg("ftol_slsqp") = 1e-8,
    py::arg("block_update") = true, py::arg("coord_cycles") = 4,
    py::arg("seed") = 42,
    "Run the full DE → SLSQP → block-update extraction pipeline in C++");


    m.def("get_num_threads", []() {
        int n = 1;
#ifdef _OPENMP
        n = omp_get_max_threads();
#endif
        return n;
    }, "Return the number of OpenMP threads available");

    m.attr("__version__") = "1.0.0";
}
