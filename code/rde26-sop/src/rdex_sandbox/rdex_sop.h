#ifndef RDEX_SOP_H
#define RDEX_SOP_H

#include <string>
#include <utility>
#include <vector>

struct OptimizationResultSOP {
    std::vector<double> best_solution;
    double best_fitness = 0.0;
    int evaluations_used = 0;
    bool success = false;
};

void set_cec2017_module_directory(const std::string& directory);
void set_random_seed(unsigned seed);
std::vector<std::pair<int, double>> get_last_best_trace();

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations
);

void set_memory_size(int v);
void set_eb_hybrid_rate_init(double v);
void set_original_eb_donor_enabled(bool enabled);
void set_cauchy_perturbation_prob(double v);
void set_f_sigma(double v);
void set_eb_f_cr_params(double f_cauchy_sigma,
                        double f_cauchy_fallback_mu,
                        double cr_sigma,
                        double cr_fallback_mu,
                        double cr_early_min1,
                        double cr_early_min2);
void set_eb_schedule_params(double early_disable_frac, double rand_multiplier);
void set_select_pressure_params(double exp_k, double psize_coef);
void set_weighted_crossover_enabled(bool enabled);
void set_xai_internal_enabled(bool enabled);
void set_trust_region_params(bool enabled,
                             double length_init,
                             double length_min,
                             double length_max,
                             int success_tolerance,
                             int failure_tolerance);
void set_trust_region_controller_params(int activation_mode,
                                        double late_start_frac,
                                        double mix_rate,
                                        int stagnation_generations);

#endif
