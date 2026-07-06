#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <unistd.h>  // getcwd, chdir
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "problem_bridge.h"
#include "rdex_sop.h"

using namespace std;

using PolicyControlCallbackSOP = std::function<std::vector<double>(
    const std::vector<double>&,
    const std::vector<double>&,
    const std::vector<double>&,
    const std::vector<double>&)>;
using OpenActionCallbackSOP = std::function<std::vector<double>(
    const std::vector<double>&,
    const std::vector<double>&,
    const std::vector<double>&,
    const std::vector<double>&,
    const std::vector<double>&,
    const std::vector<double>&,
    const std::vector<double>&,
    const std::vector<double>&)>;

// CEC2017 evaluator（官方 cec17_func.cpp）
// 注意：不使用 extern "C"，因为 cec17_func.cpp 以 C++ 编译。
void cec17_test_func(double* x, double* f, int nx, int mx, int func_num);
void cec17_set_input_data_dir(const char* dir);
void cec17_reset_state();

// LSRTDE-style deterministic seed.  Runs may still override this through
// set_random_seed(), but the default is no longer time-dependent.
unsigned globalseed = 2024;
unsigned seed1 = globalseed;
unsigned seed2 = globalseed + 100;
unsigned seed3 = globalseed + 200;
unsigned seed4 = globalseed + 300;
unsigned seed5 = globalseed + 400;
std::mt19937 generator_uni_i(seed1);
std::mt19937 generator_uni_r(seed2);
std::mt19937 generator_norm(seed3);
std::mt19937 generator_uni_i_3(seed4);
std::mt19937 generator_cachy(seed5);
std::uniform_int_distribution<int> uni_int(0, 32768);
std::uniform_real_distribution<double> uni_real(0.0, 1.0);
std::normal_distribution<double> norm_dist(0.0, 1.0);
std::cauchy_distribution<double> cachy_dist(0.0, 1.0);

const char* algName = "RDEx_SOP_platform";
double EB_hybrid_rate_init = 0.6;

// ====== 可调超参数（默认值保持与原始 RDEx_SOP 一致） ======
// 说明：
//   - 这些量在参考实现中为硬编码常数，这里提升为全局变量并提供 setter，
//     以便从 Python 侧（RDExSOPOptimizer）按需覆盖；
//   - 若外部未调用对应 setter，则保持原始默认值，不改变参考算法行为。
double F_sigma_global_sop = 0.02;          // meanF 周围的高斯扰动标准差
double CauchyPerturbationProb_global_sop = 0.1;  // 柯西局部扰动概率
double F_cauchy_sigma_global_sop = 0.1;    // EB 分支中 F 的柯西扰动尺度
double F_cauchy_fallback_mu_global_sop = 0.9; // EB 分支中 F 的备用均值
double Cr_sigma_global_sop = 0.1;          // EB 分支中 CR 的高斯扰动尺度
double Cr_fallback_mu_global_sop = 0.9;    // EB 分支中 CR 的备用均值
double Cr_early_min1_global_sop = 0.7;     // 早期 CR 最小值1（NFE/MaxFE<0.25）
double Cr_early_min2_global_sop = 0.6;     // 早期 CR 最小值2（NFE/MaxFE<0.5）
double EB_early_disable_frac_global_sop = 0.7; // 早期阶段抑制 EB 的 NFE/MaxFE 阈值
double EB_rand_multiplier_global_sop = 2.0;    // 早期阶段 Rand_EB 放大倍数
double SelectExp_global_sop = 7.0;         // SuccessRate 指数项（控制 psizeval 收缩）
double PsizeCoef_global_sop = 0.7;         // psizeval 系数（前沿父代窗口大小）
bool OriginalEbDonorEnabled_global_sop = false; // false keeps the current platform-modified EB donor
bool OriginalPerturbationRandEnabled_global_sop = false; // original RDEx used C rand() for this Bernoulli draw

// ====== Search trust region（默认关闭） ======
bool TrustRegionEnabled_global_sop = false;
double TrustRegionLengthInit_global_sop = 2.0;
double TrustRegionLengthMin_global_sop = std::pow(0.5, 7);
double TrustRegionLengthMax_global_sop = 2.0;
int TrustRegionSuccessTol_global_sop = 3;
int TrustRegionFailureTol_global_sop = 5;
int TrustRegionActivationMode_global_sop = 0;  // 0=always, 1=late-progress, 2=stagnation-triggered, 3=late+stagnation, 4=post-late stagnation
double TrustRegionLateStartFrac_global_sop = 0.6;
double TrustRegionMixRate_global_sop = 1.0;
int TrustRegionStagnationTol_global_sop = 5;
bool TrustRegionDeterministicMix_global_sop = false;

// ====== XAI 扩展：按维度权重偏置交叉掩码（默认关闭） ======
bool WeightedCrossoverEnabled_global_sop = false;
std::vector<double> DimensionWeights_global_sop;

// ====== XAI 内部版：C++ 内部拟合代理并更新维度权重（默认关闭） ======
bool XaiInternalEnabled_global_sop = false;
int XaiWarmupEvals_global_sop = 1000;
int XaiUpdateInterval_global_sop = 500;
int XaiWindowSize_global_sop = 5000;
double XaiRidgeAlpha_global_sop = 1e-2;
// importance_mode: 0=abs(beta) 1=PFI(MSE増分) 2=SHAP(best sample, linear)
int XaiImportanceMode_global_sop = 1;
bool XaiUseRankTarget_global_sop = true;
double XaiWeightFloor_global_sop = 1e-3;
double XaiWeightSmoothing_global_sop = 0.8;
double XaiTemperature_global_sop = 1.0;

// 初期値設定（ICE/PDP ライク）：0=なし 1=ICE(best) 2=PDP(top-k mean)
int XaiInitMode_global_sop = 0;
int XaiInitPhaseEvals_global_sop = 0;
double XaiInitProb_global_sop = 1.0;
int XaiInitTopK_global_sop = 5;
std::vector<double> XaiICEValues_global_sop;
std::vector<double> XaiPDPValues_global_sop;

// ====== Initial RBF surrogate injection（默认关闭） ======
struct RBFModelSOP {
    int dim = 0;
    int sample_count = 0;
    double y_mean = 0.0;
    double y_scale = 1.0;
    std::vector<double> samples;  // row-major, normalized to [-1, 1] for CEC bounds
    std::vector<double> lambda;
    std::vector<double> gamma;
};

static bool RbfInitialInjectionEnabled_global_sop = false;
static bool RbfRecordTrueSamples_global_sop = false;
static bool RbfSurrogateEvalMode_global_sop = false;
static bool RbfFastSubsetSelection_global_sop = true;
static int RbfSampleCap_global_sop = 300;
static int RbfSurrogatePatience_global_sop = 30;
static int RbfLocalPoolCap_global_sop = 1200;
static RBFModelSOP* ActiveRBFModel_global_sop = nullptr;
static double RbfSurrogateBestSeen_global_sop = std::numeric_limits<double>::infinity();
static int RbfNoImproveGenerations_global_sop = 0;
static std::vector<double> RbfArchiveSamples_global_sop;  // row-major normalized true samples
static std::vector<double> RbfArchiveFitness_global_sop;
static std::vector<double> LastRBFDiagnostics_global_sop(8, 0.0);

// 内部学習用リングバッファ（flattened）
static std::vector<double> XaiXBuf;
static std::vector<double> XaiYBuf;
static int XaiBufPos = 0;
static int XaiBufCount = 0;
static int XaiBufNVars = 0;

// ====== History Data Collection Engine ======
static bool HistoryCollectionEnabled_global_sop = false;
static double HistoryArchiveDMin_global_sop = 0.5; // Default distance threshold
static std::vector<double> g_diverse_archive_x;
static std::vector<double> g_diverse_archive_f;

static std::vector<double> g_trajectory_x;
static std::vector<double> g_trajectory_f;
static std::vector<int> g_trajectory_fe;

static std::vector<double> g_snapshot_x;
static std::vector<double> g_snapshot_f;
static int g_snapshot_fe = -1;

static std::vector<double> g_true_optimum_x;
// History NN Archive
static std::vector<double> g_nn_archive_x;
static std::vector<double> g_nn_archive_f;
static std::vector<double> g_nn_archive_dist;

// ====== Imitation-learning expert trajectory logger ======
static bool ImitationLoggingEnabled_global_sop = false;
static int ImitationMaxExamples_global_sop = 0;
static int ImitationStride_global_sop = 1;
static long long ImitationSeenUpdates_global_sop = 0;

static const std::vector<std::string> ImitationStatusNames_global_sop = {
    "eval_progress",
    "generation_progress",
    "front_size_frac",
    "success_rate",
    "eb_hybrid_rate",
    "mean_f",
    "sigma_f",
    "memory_f_mean",
    "memory_f_std",
    "memory_cr_mean",
    "memory_cr_std",
    "front_best_slog",
    "front_mean_slog",
    "front_std_log",
    "front_median_slog",
    "front_worst_slog",
    "target_fit_slog",
    "target_rank_frac",
    "psize_frac",
    "psize2_frac",
    "eval_left_frac",
    "dim_log",
    "bound_span_log",
    "front_diversity_mean",
    "front_diversity_std",
    "best_to_worst_distance",
};

static const std::vector<std::string> ImitationRoleNames_global_sop = {
    "target_front",
    "target_pop_index",
    "pbest_window_prand",
    "gbest",
    "gworst",
    "front_median",
    "front_rand1",
    "pop_rand2",
    "eb_order_best",
    "eb_order_medium",
    "eb_order_worst",
    "archive_best",
    "archive_median",
    "archive_worst",
};

static const std::vector<std::string> ImitationActionMetaNames_global_sop = {
    "f",
    "cr",
    "actual_cr",
    "use_eb",
    "perturbation",
    "use_trust_region",
    "parent_fit_slog",
    "trial_fit_slog",
    "improvement_slog",
    "success",
};

static const std::vector<std::string> ImitationIntMetaNames_global_sop = {
    "func_id",
    "generation",
    "nfe_before",
    "chosen_index",
    "prand_index",
    "rand1_index",
    "rand2_index",
    "will_crossover",
};

static const std::vector<std::string> ImitationMechanismMetaNames_global_sop = {
    "memory_current_index",
    "memory_current_index2",
    "rand_eb",
    "eb_decision_threshold",
    "normal_f_mu",
    "normal_f_sigma",
    "normal_cr_mu",
    "normal_cr_sigma",
    "eb_f_mu",
    "eb_f_sigma",
    "eb_cr_mu",
    "eb_cr_sigma",
    "psizeval",
    "psizeval2",
    "will_crossover_norm",
    "progress",
    "normal_f_noise",
    "normal_f_rejections",
    "normal_cr_noise",
    "eb_f_noise",
    "eb_f_rejections",
    "eb_cr_noise",
    "perturbation_uniform",
};

static const std::vector<std::string> ImitationVectorMaterialNames_global_sop = {
    "crossover_uniform",
    "perturbation_candidate_delta",
    "repair_candidate_delta",
    "repair_applied",
    "rdex_precallback_trial_delta",
    "rdex_precallback_trial_x",
    "perturbation_cauchy_noise",
    "pre_repair_trial_delta",
    "pre_repair_trial_x",
};

static std::vector<double> g_imitation_status;
static std::vector<double> g_imitation_roles;
static std::vector<double> g_imitation_role_fitness;
static std::vector<double> g_imitation_mask;
static std::vector<double> g_imitation_delta;
static std::vector<double> g_imitation_trial_x;
static std::vector<double> g_imitation_action_meta;
static std::vector<int> g_imitation_int_meta;
static std::vector<double> g_imitation_memory_f;
static std::vector<double> g_imitation_memory_cr;
static std::vector<double> g_imitation_mechanism_meta;
static std::vector<double> g_imitation_vector_materials;

static PolicyControlCallbackSOP PolicyControlCallback_global_sop;
static bool PolicyControlCallbackEnabled_global_sop = false;
static OpenActionCallbackSOP OpenActionCallback_global_sop;
static bool OpenActionCallbackEnabled_global_sop = false;

// Custom Initial Population
std::vector<double> g_custom_init_pop;
void set_custom_initial_population(const std::vector<double>& pop_x_flat) {
    g_custom_init_pop = pop_x_flat;
}

// ============================================


// 全局评估相关变量（与参考版本保持命名一致）
const int ResTsize2 = 1001;  // 记录长度（保留接口，不强制使用）
int stepsFEval[ResTsize2 - 1];
double ResultsArray[ResTsize2];
int LastFEcount = 0;
int NFEval = 0;
int MaxFEval = 0;
int GNVars = 0;
double tempF[1];
double globalbest = 0.0;
bool globalbestinit = false;

// 平台传入的边界（标量包络）
double Left_platform = -100.0;
double Right_platform = 100.0;

// ====== Pure C++ evaluator mode (CEC2017-SOP) ======
static bool UseCec2017Evaluator_global_sop = false;
static std::string Cec2017ModuleDirectory_global_sop;
static int ParallelWorkers_global_sop = 0;

void configure_openmp_runtime_once() {
#ifdef _OPENMP
    static bool configured = false;
    if (!configured) {
        omp_set_dynamic(0);
        configured = true;
    }
#endif
}

class WorkingDirectoryGuard {
private:
    std::string original_dir;
    bool changed = false;

public:
    explicit WorkingDirectoryGuard(const std::string& target_dir) {
        if (target_dir.empty()) return;
        char cwd[4096];
        if (getcwd(cwd, sizeof(cwd)) != nullptr) {
            original_dir = cwd;
            if (chdir(target_dir.c_str()) == 0) {
                changed = true;
            }
        }
    }

    ~WorkingDirectoryGuard() {
        if (changed && !original_dir.empty()) {
            (void)chdir(original_dir.c_str());
        }
    }
};

class Cec2017EvalModeGuard {
private:
    bool old_mode;

public:
    explicit Cec2017EvalModeGuard(bool new_mode) : old_mode(UseCec2017Evaluator_global_sop) {
        UseCec2017Evaluator_global_sop = new_mode;
    }

    ~Cec2017EvalModeGuard() {
        UseCec2017Evaluator_global_sop = old_mode;
    }
};

// ====== 工具函数 ======

int IntRandom(int target) {
    if (target == 0) return 0;
    return uni_int(generator_uni_i) % target;
}

double Random(double minimal, double maximal) {
    return uni_real(generator_uni_r) * (maximal - minimal) + minimal;
}

double NormRand(double mu, double sigma) {
    return norm_dist(generator_norm) * sigma + mu;
}

double NormRandWithNoise(double mu, double sigma, double& noise) {
    noise = norm_dist(generator_norm);
    return noise * sigma + mu;
}

double CachyRand(double mu, double sigma) {
    return cachy_dist(generator_cachy) * sigma + mu;
}

double CachyRandWithNoise(double mu, double sigma, double& noise) {
    noise = cachy_dist(generator_cachy);
    return noise * sigma + mu;
}

bool LocalPerturbationEnabled() {
    if (OriginalPerturbationRandEnabled_global_sop) {
        return rand() / (double)RAND_MAX < CauchyPerturbationProb_global_sop;
    }
    return Random(0.0, 1.0) < CauchyPerturbationProb_global_sop;
}

bool LocalPerturbationEnabledWithUniform(double& u) {
    if (OriginalPerturbationRandEnabled_global_sop) {
        u = rand() / (double)RAND_MAX;
    } else {
        u = Random(0.0, 1.0);
    }
    return u < CauchyPerturbationProb_global_sop;
}

static inline double tr_normalize01(double x, double lb, double ub) {
    if (!(ub > lb)) return 0.5;
    return (x - lb) / (ub - lb);
}

static inline double tr_denormalize01(double z, double lb, double ub) {
    if (!(ub > lb)) return lb;
    return lb + z * (ub - lb);
}

double EbDonorValue(double base,
                    double best,
                    double medium,
                    double worst,
                    double rand1,
                    double rand2,
                    double F) {
    if (OriginalEbDonorEnabled_global_sop) {
        return base + F * (best - base) + F * (medium - worst);
    }
    return base + F * (best - base) + 0.9 * F * (medium - worst) +
           0.1 * (1 - NFEval / MaxFEval) * (rand1 - rand2);
}

void qSort2int(double* Mass, int* Mass2, int low, int high) {
    int i = low;
    int j = high;
    double x = Mass[(low + high) >> 1];
    do {
        while (Mass[i] < x) ++i;
        while (Mass[j] > x) --j;
        if (i <= j) {
            double temp = Mass[i];
            Mass[i] = Mass[j];
            Mass[j] = temp;
            int temp2 = Mass2[i];
            Mass2[i] = Mass2[j];
            Mass2[j] = temp2;
            i++;
            j--;
        }
    } while (i <= j);
    if (low < j) qSort2int(Mass, Mass2, low, j);
    if (i < high) qSort2int(Mass, Mass2, i, high);
}

// 参考代码中的 SaveBestValues 仅用于生成 Min_EV 轨迹，
// 对算法行为无影响；平台版保留一个安全实现以备后续扩展。
void SaveBestValues(int /*func_index*/) {
    double temp = globalbest;
    if (temp <= 1E-8 && ResultsArray[ResTsize2 - 1] == MaxFEval)
        ResultsArray[ResTsize2 - 1] = NFEval;
    for (int stepFEcount = LastFEcount; stepFEcount < ResTsize2 - 1; stepFEcount++) {
        if (NFEval >= stepsFEval[stepFEcount]) {
            if (temp <= 1E-8) temp = 0;
            ResultsArray[stepFEcount] = temp;
            LastFEcount = stepFEcount + 1;
        }
    }
}

int resolve_parallel_workers(int requested, int work_items) {
    configure_openmp_runtime_once();
    int workers = requested;
    if (workers <= 0) {
#ifdef _OPENMP
        workers = omp_get_max_threads();
#else
        workers = 1;
#endif
    }
    if (workers < 1) workers = 1;
    if (work_items > 0 && workers > work_items) workers = work_items;
    return workers;
}

double evaluate_cec2017_error_no_count(double* x, int dim, int func_num) {
    double raw = 0.0;
    cec17_test_func(x, &raw, dim, 1, int(func_num));
    const double optimal_value = double(func_num) * 100.0;
    double err = raw - optimal_value;
    if (std::fabs(err) <= 1e-8) err = 0.0;
    return err;
}

void evaluate_cec2017_rows_parallel(double** rows,
                                    int count,
                                    int dim,
                                    int func_num,
                                    double* out,
                                    int requested_workers) {
    const int workers = resolve_parallel_workers(requested_workers, count);
    if (workers <= 1) {
        for (int i = 0; i < count; ++i) {
            out[i] = evaluate_cec2017_error_no_count(rows[i], dim, func_num);
        }
        return;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(workers)
#endif
    for (int i = 0; i < count; ++i) {
        out[i] = evaluate_cec2017_error_no_count(rows[i], dim, func_num);
    }
}

void evaluate_cec2017_flat_parallel(std::vector<double>& rows,
                                    int count,
                                    int dim,
                                    int func_num,
                                    std::vector<double>& out,
                                    int requested_workers) {
    out.resize(size_t(count));
    const int workers = resolve_parallel_workers(requested_workers, count);
    if (workers <= 1) {
        for (int i = 0; i < count; ++i) {
            out[size_t(i)] = evaluate_cec2017_error_no_count(
                &rows[size_t(i) * size_t(dim)], dim, func_num);
        }
        return;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(workers)
#endif
    for (int i = 0; i < count; ++i) {
        out[size_t(i)] = evaluate_cec2017_error_no_count(
            &rows[size_t(i) * size_t(dim)], dim, func_num);
    }
}

int choose_generation_workers(int requested_workers,
                              int work_items,
                              int small_batch_threshold) {
    int workers = resolve_parallel_workers(requested_workers, work_items);
    if (small_batch_threshold <= 0) {
        return workers;
    }
    if (small_batch_threshold > 0 && work_items <= small_batch_threshold) {
        return 1;
    }
    if (workers > 1 && work_items > 0 && work_items < workers * 2) {
        workers = std::max(1, work_items / 2);
    }
    return resolve_parallel_workers(workers, work_items);
}

static void rbf_reset_archive() {
    RbfArchiveSamples_global_sop.clear();
    RbfArchiveFitness_global_sop.clear();
    std::fill(LastRBFDiagnostics_global_sop.begin(), LastRBFDiagnostics_global_sop.end(), 0.0);
}

static int rbf_archive_count(int dim) {
    (void)dim;
    return int(RbfArchiveFitness_global_sop.size());
}

static const double* rbf_archive_point_ptr(int index, int dim) {
    return &RbfArchiveSamples_global_sop[size_t(index) * size_t(dim)];
}

static void rbf_record_true_sample(const double* point, int dim, double fitness) {
    for (int j = 0; j < dim; ++j) {
        RbfArchiveSamples_global_sop.push_back(point[j] / 100.0);
    }
    RbfArchiveFitness_global_sop.push_back(fitness);
}

static std::vector<double> rbf_normalize_point(const double* point, int dim) {
    std::vector<double> normalized(size_t(dim), 0.0);
    for (int j = 0; j < dim; ++j) {
        normalized[size_t(j)] = point[j] / 100.0;
    }
    return normalized;
}

static double rbf_distance_normalized_to_archive(const std::vector<double>& center,
                                                 int archive_index,
                                                 int dim) {
    const double* point = rbf_archive_point_ptr(archive_index, dim);
    double sum = 0.0;
    for (int j = 0; j < dim; ++j) {
        const double diff = center[size_t(j)] - point[j];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

static double rbf_distance_archive_archive(int lhs, int rhs, int dim) {
    const double* a = rbf_archive_point_ptr(lhs, dim);
    const double* b = rbf_archive_point_ptr(rhs, dim);
    double sum = 0.0;
    for (int j = 0; j < dim; ++j) {
        const double diff = a[j] - b[j];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

static double rbf_kernel_cubic_flat(const std::vector<double>& normalized,
                                   const RBFModelSOP& model,
                                   int sample_index) {
    const double* sample = &model.samples[size_t(sample_index) * size_t(model.dim)];
    double sum = 0.0;
    for (int j = 0; j < model.dim; ++j) {
        const double diff = normalized[size_t(j)] - sample[j];
        sum += diff * diff;
    }
    const double distance = std::sqrt(sum);
    return distance * distance * distance;
}

static double rbf_kernel_cubic_sample_sample(const RBFModelSOP& model,
                                            int lhs,
                                            int rhs) {
    const double* a = &model.samples[size_t(lhs) * size_t(model.dim)];
    const double* b = &model.samples[size_t(rhs) * size_t(model.dim)];
    double sum = 0.0;
    for (int j = 0; j < model.dim; ++j) {
        const double diff = a[j] - b[j];
        sum += diff * diff;
    }
    const double distance = std::sqrt(sum);
    return distance * distance * distance;
}

static double rbf_evaluate_point(const double* point, const RBFModelSOP& model) {
    std::vector<double> normalized = rbf_normalize_point(point, model.dim);
    double radial = 0.0;
    for (int i = 0; i < model.sample_count; ++i) {
        radial += model.lambda[size_t(i)] * rbf_kernel_cubic_flat(normalized, model, i);
    }
    double tail = model.gamma.back();
    for (int j = 0; j < model.dim; ++j) {
        tail += model.gamma[size_t(j)] * normalized[size_t(j)];
    }
    return (radial + tail) * model.y_scale + model.y_mean;
}

static void record_history_data(const double* x, int dim, double fit) {
    if (!HistoryCollectionEnabled_global_sop) return;
    
    // 1. Trajectory Log
    if (g_trajectory_f.empty() || fit < g_trajectory_f.back()) {
        g_trajectory_f.push_back(fit);
        g_trajectory_fe.push_back(NFEval);
        for (int i = 0; i < dim; ++i) g_trajectory_x.push_back(x[i]);
    }
    
    // 2. NN Archive
    if (!g_true_optimum_x.empty() && (int)g_true_optimum_x.size() == dim) {
        double dist_sq = 0.0;
        for (int i = 0; i < dim; ++i) {
            double d = x[i] - g_true_optimum_x[i];
            dist_sq += d * d;
        }
        double dist = std::sqrt(dist_sq);
        
        if (g_nn_archive_f.size() < 1000) {
            g_nn_archive_f.push_back(fit);
            g_nn_archive_dist.push_back(dist);
            for (int i = 0; i < dim; ++i) g_nn_archive_x.push_back(x[i]);
        } else {
            // Find max distance in NN archive
            double max_dist = -1.0;
            int max_idx = -1;
            for (int i = 0; i < 1000; ++i) {
                if (g_nn_archive_dist[i] > max_dist) {
                    max_dist = g_nn_archive_dist[i];
                    max_idx = i;
                }
            }
            if (dist < max_dist) {
                g_nn_archive_f[max_idx] = fit;
                g_nn_archive_dist[max_idx] = dist;
                for (int i = 0; i < dim; ++i) g_nn_archive_x[max_idx * dim + i] = x[i];
            }
        }
    }
    
    // 3. Spatially Diverse Top-K Archive
    const int K = 2000;
    const double D_min = HistoryArchiveDMin_global_sop;
    
    int replace_idx = -1;
    bool too_close = false;
    
    int current_size = g_diverse_archive_f.size();
    for (int i = 0; i < current_size; ++i) {
        double dist_sq = 0.0;
        const double* old_x = &g_diverse_archive_x[i * dim];
        for (int j = 0; j < dim; ++j) {
            double d = x[j] - old_x[j];
            dist_sq += d * d;
        }
        double dist = std::sqrt(dist_sq);
        
        if (dist < D_min) {
            too_close = true;
            if (fit < g_diverse_archive_f[i]) {
                replace_idx = i;
            }
            break;
        }
    }
    
    if (too_close) {
        if (replace_idx >= 0) {
            g_diverse_archive_f[replace_idx] = fit;
            for (int i = 0; i < dim; ++i) g_diverse_archive_x[replace_idx * dim + i] = x[i];
        }
    } else {
        if (current_size < K) {
            g_diverse_archive_f.push_back(fit);
            for (int i = 0; i < dim; ++i) g_diverse_archive_x.push_back(x[i]);
        } else {
            double max_f = -1e100;
            int max_idx = -1;
            for (int i = 0; i < K; ++i) {
                if (g_diverse_archive_f[i] > max_f) {
                    max_f = g_diverse_archive_f[i];
                    max_idx = i;
                }
            }
            if (fit < max_f) {
                g_diverse_archive_f[max_idx] = fit;
                for (int i = 0; i < dim; ++i) g_diverse_archive_x[max_idx * dim + i] = x[i];
            }
        }
    }
}

// 平台版目标函数：通过 problem_bridge 回调 Python Problem.evaluate。
double cec_24(double* HostVector, int func_num) {
    if (RbfSurrogateEvalMode_global_sop) {
        if (ActiveRBFModel_global_sop == nullptr) {
            throw std::runtime_error("RBF surrogate evaluator is active without a model");
        }
        tempF[0] = rbf_evaluate_point(HostVector, *ActiveRBFModel_global_sop);
        NFEval++;
        return tempF[0];
    }

    if (UseCec2017Evaluator_global_sop) {
        // 创建输入副本（官方 evaluator 可能会修改输入；与 cec2017_cpp_module 行为对齐）
        thread_local std::vector<double> x_copy;
        x_copy.resize(size_t(GNVars));
        for (int j = 0; j < GNVars; ++j) {
            x_copy[size_t(j)] = HostVector[j];
        }

        tempF[0] = evaluate_cec2017_error_no_count(x_copy.data(), GNVars, int(func_num));
    } else {
        tempF[0] = problem_bridge::evaluate_solution(HostVector, GNVars);
    }
    NFEval++;
    if (RbfRecordTrueSamples_global_sop) {
        rbf_record_true_sample(HostVector, GNVars, tempF[0]);
    }
    record_history_data(HostVector, GNVars, tempF[0]);
    return tempF[0];
}

static void xai_reset_buffers(int nvars) {
    XaiBufNVars = nvars;
    int win = XaiWindowSize_global_sop;
    if (win < 10) win = 10;
    if (win > 200000) win = 200000;
    XaiXBuf.assign(size_t(win) * size_t(nvars), 0.0);
    XaiYBuf.assign(size_t(win), std::numeric_limits<double>::quiet_NaN());
    XaiBufPos = 0;
    XaiBufCount = 0;
    XaiICEValues_global_sop.assign(nvars, 0.0);
    XaiPDPValues_global_sop.assign(nvars, 0.0);
    if ((int)DimensionWeights_global_sop.size() != nvars) {
        DimensionWeights_global_sop.assign(nvars, 1.0 / std::max(1, nvars));
    }
}

static void xai_push_sample(const double* x, double y, int nvars) {
    if (XaiBufNVars != nvars || XaiXBuf.empty() || XaiYBuf.empty()) return;
    const int win = (int)XaiYBuf.size();
    const int pos = XaiBufPos;
    double* row = &XaiXBuf[size_t(pos) * size_t(nvars)];
    for (int j = 0; j < nvars; ++j) row[j] = x[j];
    XaiYBuf[pos] = y;
    XaiBufPos = (pos + 1) % win;
    XaiBufCount = std::min(XaiBufCount + 1, win);
}

static bool solve_linear_system(std::vector<double>& A, std::vector<double>& b, int n) {
    // Gaussian elimination with partial pivoting: A is row-major n*n
    for (int i = 0; i < n; ++i) {
        int pivot = i;
        double best = std::fabs(A[size_t(i) * n + i]);
        for (int r = i + 1; r < n; ++r) {
            double v = std::fabs(A[size_t(r) * n + i]);
            if (v > best) {
                best = v;
                pivot = r;
            }
        }
        if (!(best > 1e-18)) return false;
        if (pivot != i) {
            for (int c = i; c < n; ++c) {
                std::swap(A[size_t(i) * n + c], A[size_t(pivot) * n + c]);
            }
            std::swap(b[i], b[pivot]);
        }
        double diag = A[size_t(i) * n + i];
        for (int r = i + 1; r < n; ++r) {
            double f = A[size_t(r) * n + i] / diag;
            if (f == 0.0) continue;
            A[size_t(r) * n + i] = 0.0;
            for (int c = i + 1; c < n; ++c) {
                A[size_t(r) * n + c] -= f * A[size_t(i) * n + c];
            }
            b[r] -= f * b[i];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int c = i + 1; c < n; ++c) {
            sum -= A[size_t(i) * n + c] * b[c];
        }
        double diag = A[size_t(i) * n + i];
        if (!(std::fabs(diag) > 1e-18)) return false;
        b[i] = sum / diag;
    }
    return true;
}

static int xai_argmin_pos(const std::vector<double>& y_raw, const std::vector<int>& pos_list) {
    int best_i = 0;
    double best_y = y_raw[0];
    for (int i = 1; i < (int)y_raw.size(); ++i) {
        if (y_raw[i] < best_y) {
            best_y = y_raw[i];
            best_i = i;
        }
    }
    return pos_list[best_i];
}

static void xai_update_ice_pdp_values(int nvars, const std::vector<double>& y_raw, const std::vector<int>& pos_list) {
    if (y_raw.empty()) return;
    int best_pos = xai_argmin_pos(y_raw, pos_list);
    const double* best_row = &XaiXBuf[size_t(best_pos) * size_t(nvars)];
    XaiICEValues_global_sop.assign(best_row, best_row + nvars);

    int topk = XaiInitTopK_global_sop;
    if (topk < 1) topk = 1;
    topk = std::min(topk, (int)y_raw.size());
    std::vector<std::pair<double, int>> ys;
    ys.reserve(y_raw.size());
    for (int i = 0; i < (int)y_raw.size(); ++i) ys.push_back({y_raw[i], pos_list[i]});
    std::nth_element(ys.begin(), ys.begin() + topk - 1, ys.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    XaiPDPValues_global_sop.assign(nvars, 0.0);
    for (int k = 0; k < topk; ++k) {
        const double* row = &XaiXBuf[size_t(ys[k].second) * size_t(nvars)];
        for (int j = 0; j < nvars; ++j) XaiPDPValues_global_sop[j] += row[j];
    }
    for (int j = 0; j < nvars; ++j) XaiPDPValues_global_sop[j] /= double(topk);
}

static void xai_maybe_update_weights(int nvars) {
    if (!XaiInternalEnabled_global_sop) return;
    if (nvars <= 0) return;
    if (NFEval < XaiWarmupEvals_global_sop) return;
    if (XaiUpdateInterval_global_sop <= 0) return;
    if (NFEval % XaiUpdateInterval_global_sop != 0) return;
    if (XaiBufNVars != nvars || XaiXBuf.empty() || XaiYBuf.empty()) return;
    const int n = XaiBufCount;
    if (n < 10) return;

    const int win = (int)XaiYBuf.size();
    std::vector<double> y_raw;
    std::vector<int> pos_list;
    y_raw.reserve((size_t)n);
    pos_list.reserve((size_t)n);
    for (int i = 0; i < n; ++i) {
        int pos = (XaiBufPos - n + i);
        pos %= win;
        if (pos < 0) pos += win;
        double y = XaiYBuf[pos];
        if (!std::isfinite(y)) continue;
        y_raw.push_back(y);
        pos_list.push_back(pos);
    }
    const int n_eff = (int)y_raw.size();
    if (n_eff < 10) return;

    // Compute mean/std for each dimension on raw X
    std::vector<double> x_mean((size_t)nvars, 0.0);
    std::vector<double> x_var((size_t)nvars, 0.0);
    for (int i = 0; i < n_eff; ++i) {
        const double* row = &XaiXBuf[size_t(pos_list[i]) * size_t(nvars)];
        for (int j = 0; j < nvars; ++j) x_mean[j] += row[j];
    }
    for (int j = 0; j < nvars; ++j) x_mean[j] /= double(n_eff);
    for (int i = 0; i < n_eff; ++i) {
        const double* row = &XaiXBuf[size_t(pos_list[i]) * size_t(nvars)];
        for (int j = 0; j < nvars; ++j) {
            double d = row[j] - x_mean[j];
            x_var[j] += d * d;
        }
    }
    std::vector<double> x_std((size_t)nvars, 1.0);
    for (int j = 0; j < nvars; ++j) {
        double v = x_var[j] / double(std::max(1, n_eff));
        x_std[j] = (v > 0.0) ? std::sqrt(v) : 1.0;
    }

    // Target y: rank or raw, then standardize
    std::vector<double> y((size_t)n_eff, 0.0);
    if (XaiUseRankTarget_global_sop) {
        std::vector<int> idx(n_eff);
        for (int i = 0; i < n_eff; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return y_raw[a] < y_raw[b]; });
        for (int r = 0; r < n_eff; ++r) y[idx[r]] = double(r);
    } else {
        for (int i = 0; i < n_eff; ++i) y[i] = y_raw[i];
    }
    double y_mean = 0.0;
    for (int i = 0; i < n_eff; ++i) y_mean += y[i];
    y_mean /= double(n_eff);
    double y_var = 0.0;
    for (int i = 0; i < n_eff; ++i) {
        double d = y[i] - y_mean;
        y_var += d * d;
    }
    double y_std = (y_var / double(std::max(1, n_eff)) > 0.0)
        ? std::sqrt(y_var / double(std::max(1, n_eff)))
        : 1.0;
    if (!(y_std > 0.0)) y_std = 1.0;
    for (int i = 0; i < n_eff; ++i) y[i] = (y[i] - y_mean) / y_std;

    // Build standardized X matrix (n_eff x nvars)
    std::vector<double> Xs(size_t(n_eff) * size_t(nvars), 0.0);
    for (int i = 0; i < n_eff; ++i) {
        const double* row = &XaiXBuf[size_t(pos_list[i]) * size_t(nvars)];
        double* xs = &Xs[size_t(i) * size_t(nvars)];
        for (int j = 0; j < nvars; ++j) xs[j] = (row[j] - x_mean[j]) / x_std[j];
    }

    // Ridge fit on standardized data
    std::vector<double> XtX(size_t(nvars) * size_t(nvars), 0.0);
    std::vector<double> Xty((size_t)nvars, 0.0);
    for (int i = 0; i < n_eff; ++i) {
        const double* xs = &Xs[size_t(i) * size_t(nvars)];
        for (int a = 0; a < nvars; ++a) Xty[a] += xs[a] * y[i];
        for (int a = 0; a < nvars; ++a) {
            double va = xs[a];
            for (int b = 0; b < nvars; ++b) {
                XtX[size_t(a) * size_t(nvars) + b] += va * xs[b];
            }
        }
    }
    double alpha = XaiRidgeAlpha_global_sop;
    if (alpha < 0.0) alpha = 0.0;
    for (int d = 0; d < nvars; ++d) XtX[size_t(d) * size_t(nvars) + d] += alpha;

    std::vector<double> beta = Xty;
    if (!solve_linear_system(XtX, beta, nvars)) return;

    std::vector<double> imp((size_t)nvars, 0.0);
    if (XaiImportanceMode_global_sop == 2) {
        // SHAP-like (linear): contribution magnitude at best raw y sample
        int best_i = 0;
        double best_y = y_raw[0];
        for (int i = 1; i < n_eff; ++i) {
            if (y_raw[i] < best_y) {
                best_y = y_raw[i];
                best_i = i;
            }
        }
        const double* xs = &Xs[size_t(best_i) * size_t(nvars)];
        for (int j = 0; j < nvars; ++j) imp[j] = std::fabs(beta[j] * xs[j]);
    } else if (XaiImportanceMode_global_sop == 1) {
        // PFI on standardized data with MSE increase; use one shared permutation
        std::vector<int> perm(n_eff);
        for (int i = 0; i < n_eff; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), generator_uni_i_3);

        std::vector<double> pred_base(n_eff, 0.0);
        double mse_base = 0.0;
        for (int i = 0; i < n_eff; ++i) {
            const double* xs = &Xs[size_t(i) * size_t(nvars)];
            double p = 0.0;
            for (int j = 0; j < nvars; ++j) p += beta[j] * xs[j];
            pred_base[i] = p;
            double e = p - y[i];
            mse_base += e * e;
        }
        mse_base /= double(n_eff);

        for (int j = 0; j < nvars; ++j) {
            double mse = 0.0;
            for (int i = 0; i < n_eff; ++i) {
                const double* xs_i = &Xs[size_t(i) * size_t(nvars)];
                const double* xs_p = &Xs[size_t(perm[i]) * size_t(nvars)];
                double p = pred_base[i] + beta[j] * (xs_p[j] - xs_i[j]);
                double e = p - y[i];
                mse += e * e;
            }
            mse /= double(n_eff);
            double delta = mse - mse_base;
            if (!std::isfinite(delta) || delta < 0.0) delta = 0.0;
            imp[j] = delta;
        }
    } else {
        for (int j = 0; j < nvars; ++j) imp[j] = std::fabs(beta[j]);
    }

    double temp = XaiTemperature_global_sop;
    if (!(temp > 0.0)) temp = 1.0;
    double floor = XaiWeightFloor_global_sop;
    if (floor < 0.0) floor = 0.0;

    std::vector<double> w_new((size_t)nvars, 0.0);
    double sumw = 0.0;
    for (int j = 0; j < nvars; ++j) {
        double v = imp[j];
        if (!std::isfinite(v) || v < 0.0) v = 0.0;
        v = std::pow(v + 1e-12, temp);
        v += floor;
        w_new[j] = v;
        sumw += v;
    }
    if (!(sumw > 0.0)) return;
    for (int j = 0; j < nvars; ++j) w_new[j] /= sumw;

    if ((int)DimensionWeights_global_sop.size() != nvars) {
        DimensionWeights_global_sop.assign(nvars, 1.0 / std::max(1, nvars));
    }
    double b = XaiWeightSmoothing_global_sop;
    if (b < 0.0) b = 0.0;
    if (b > 1.0) b = 1.0;
    std::vector<double> w((size_t)nvars, 0.0);
    double s2 = 0.0;
    for (int j = 0; j < nvars; ++j) {
        w[j] = b * DimensionWeights_global_sop[j] + (1.0 - b) * w_new[j];
        s2 += w[j];
    }
    if (s2 > 0.0) {
        for (int j = 0; j < nvars; ++j) w[j] /= s2;
        DimensionWeights_global_sop = w;
        WeightedCrossoverEnabled_global_sop = true;
    }

    xai_update_ice_pdp_values(nvars, y_raw, pos_list);
}

static void xai_apply_init_guidance(double* trial, int nvars, double Left, double Right) {
    if (!XaiInternalEnabled_global_sop) return;
    if (XaiInitMode_global_sop == 0) return;
    if (XaiInitPhaseEvals_global_sop <= 0) return;
    if (NFEval < XaiWarmupEvals_global_sop) return;
    if (NFEval > (XaiWarmupEvals_global_sop + XaiInitPhaseEvals_global_sop)) return;
    if (XaiInitProb_global_sop < 1.0 && Random(0, 1) > XaiInitProb_global_sop) return;
    if ((int)DimensionWeights_global_sop.size() != nvars) return;

    int best_j = 0;
    double best_w = DimensionWeights_global_sop[0];
    for (int j = 1; j < nvars; ++j) {
        if (DimensionWeights_global_sop[j] > best_w) {
            best_w = DimensionWeights_global_sop[j];
            best_j = j;
        }
    }

    double v = trial[best_j];
    if (XaiInitMode_global_sop == 1 && (int)XaiICEValues_global_sop.size() == nvars) {
        v = XaiICEValues_global_sop[best_j];
    } else if (XaiInitMode_global_sop == 2 && (int)XaiPDPValues_global_sop.size() == nvars) {
        v = XaiPDPValues_global_sop[best_j];
    } else {
        return;
    }

    trial[best_j] = v;
    if (trial[best_j] < Left) trial[best_j] = Random(Left, Right);
    if (trial[best_j] > Right) trial[best_j] = Random(Left, Right);
}

class Optimizer {
public:
    int MemorySize;
    int MemoryIter;
    int SuccessFilled;
    int MemoryCurrentIndex;
    int MemoryCurrentIndex2;
    int NVars;
    int NIndsCurrent;
    int NIndsFront;
    int NIndsFrontMax;
    int newNIndsFront;
    int PopulSize;
    int func_num;
    int func_index;
    int TheChosenOne;
    int Generation;
    int PFIndex;

    double bestfit;
    double SuccessRate;
    double F;
    double Cr;
    double Right;
    double Left;

    double** Popul;
    double** PopulFront;
    double** PopulTemp;
    double* FitArr;
    double* FitArrCopy;
    double* FitArrFront;
    double* Trial;
    double* tempSuccessCr;
    double* tempSuccessF;
    double* MemoryCr;
    double* MemoryF;
    double* FitDelta;
    double* Weights;

    int* Indices;
    int* Indices2;

    double* ord_best_arch;
    double* ord_medium_arch;
    double* ord_worst_arch;
    double* ord_best_popul;
    double* ord_medium_popul;
    double* ord_worst_popul;
    double* FitMass;
    double EB_hybrid_rate;
    double* FitTemp;
    double* EB_hybrid_flag;

    double* BestInd;
    bool TrustRegionEnabled;
    double TrustRegionLength;
    int TrustRegionSuccessCount;
    int TrustRegionFailureCount;
    int TrustRegionStagnationCount;
    bool TrustRegionGenerationActive;
    bool TrustRegionPreviousGenerationActive;
    std::vector<double> TrustRegionLower;
    std::vector<double> TrustRegionUpper;

    void EB_order(int prand, int Rand1, int Rand2);
    void UpdateEB_hybrid_param(double* EB_hybrid_flag, double* FitArrFront, vector<double> FitTemp);
    void PrepareTrustRegionGeneration();
    void RefreshTrustRegionBox();
    bool TrialUsesTrustRegion() const;
    void RepairTrialInTrustRegion();
    void UpdateTrustRegionAfterGeneration(double generation_best_before);

    void Initialize(int _newNInds, int _newNVars, int _newfunc_num, int _newfunc_index);
    void Clean();
    void MainCycle();
    void MainCycleSyncParallel(int parallel_workers, int small_batch_threshold = 0);
    void FindNSaveBest(bool init, int IndIter);
    void UpdateMemoryCr();
    double MeanWL(double* Vector, double* TempWeights);
    void RemoveWorst(int NInds, int NewNInds);
};

static bool (*GenerationHook_global_sop)(Optimizer*) = nullptr;

static double imitation_slog(double x) {
    if (!std::isfinite(x)) return 0.0;
    return (x >= 0.0 ? 1.0 : -1.0) * std::log1p(std::fabs(x));
}

static double imitation_mean(const double* values, int count) {
    if (count <= 0) return 0.0;
    double s = 0.0;
    for (int i = 0; i < count; ++i) s += values[i];
    return s / double(count);
}

static double imitation_std(const double* values, int count, double mean) {
    if (count <= 0) return 0.0;
    double s = 0.0;
    for (int i = 0; i < count; ++i) {
        const double d = values[i] - mean;
        s += d * d;
    }
    return std::sqrt(s / double(count));
}

static double imitation_point_distance(const double* a, const double* b, int dim) {
    double s = 0.0;
    for (int j = 0; j < dim; ++j) {
        const double d = a[j] - b[j];
        s += d * d;
    }
    return std::sqrt(s);
}

static void imitation_push_point(const double* x, int dim) {
    for (int j = 0; j < dim; ++j) {
        g_imitation_roles.push_back(x[j]);
    }
}

static void imitation_push_role(const double* x, double fit, int dim) {
    imitation_push_point(x, dim);
    g_imitation_role_fitness.push_back(imitation_slog(fit));
}

static void imitation_selected_order(const double* x1,
                                     double f1,
                                     const double* x2,
                                     double f2,
                                     const double* x3,
                                     double f3,
                                     std::vector<const double*>& xs,
                                     std::vector<double>& fs) {
    xs = {x1, x2, x3};
    fs = {f1, f2, f3};
    std::vector<int> order = {0, 1, 2};
    std::sort(order.begin(), order.end(), [&](int a, int b) { return fs[size_t(a)] < fs[size_t(b)]; });
    std::vector<const double*> xs_sorted = {
        xs[size_t(order[0])],
        xs[size_t(order[1])],
        xs[size_t(order[2])],
    };
    std::vector<double> fs_sorted = {
        fs[size_t(order[0])],
        fs[size_t(order[1])],
        fs[size_t(order[2])],
    };
    xs.swap(xs_sorted);
    fs.swap(fs_sorted);
}

static void imitation_append_point(std::vector<double>& roles, const double* x, int dim) {
    for (int j = 0; j < dim; ++j) {
        roles.push_back(x[j]);
    }
}

static void imitation_append_role(std::vector<double>& roles,
                                  std::vector<double>& role_fitness,
                                  const double* x,
                                  double fit,
                                  int dim) {
    imitation_append_point(roles, x, dim);
    role_fitness.push_back(imitation_slog(fit));
}

static void build_imitation_context_vectors(Optimizer& opt,
                                            int nfe_before,
                                            int chosen,
                                            int prand,
                                            int rand1,
                                            int rand2,
                                            double mean_f,
                                            double sigma_f,
                                            int psizeval,
                                            int psizeval2,
                                            std::vector<double>& status,
                                            std::vector<double>& roles,
                                            std::vector<double>& role_fitness) {
    const int dim = opt.NVars;
    const double span = std::max(1e-12, opt.Right - opt.Left);
    const double norm_dist = span * std::sqrt(double(std::max(1, dim)));
    const double parent_fit = opt.FitArrFront[chosen];

    const double fit_mean = imitation_mean(opt.FitArrFront, opt.NIndsFront);
    const double fit_std = imitation_std(opt.FitArrFront, opt.NIndsFront, fit_mean);
    const double memory_f_mean = imitation_mean(opt.MemoryF, opt.MemorySize);
    const double memory_f_std = imitation_std(opt.MemoryF, opt.MemorySize, memory_f_mean);
    const double memory_cr_mean = imitation_mean(opt.MemoryCr, opt.MemorySize);
    const double memory_cr_std = imitation_std(opt.MemoryCr, opt.MemorySize, memory_cr_mean);
    const double progress = MaxFEval > 0 ? double(nfe_before) / double(MaxFEval) : 1.0;
    const double generation_progress =
        MaxFEval > 0 ? double(opt.Generation * std::max(1, opt.NIndsFrontMax)) / double(MaxFEval) : 0.0;

    double div_mean = 0.0;
    double div_sq_mean = 0.0;
    for (int i = 0; i < opt.NIndsFront; ++i) {
        const double d = imitation_point_distance(opt.PopulFront[i], opt.PopulFront[0], dim) / norm_dist;
        div_mean += d;
        div_sq_mean += d * d;
    }
    div_mean /= double(std::max(1, opt.NIndsFront));
    div_sq_mean /= double(std::max(1, opt.NIndsFront));
    const double div_std = std::sqrt(std::max(0.0, div_sq_mean - div_mean * div_mean));
    const double best_to_worst =
        imitation_point_distance(opt.PopulFront[0], opt.PopulFront[opt.NIndsFront - 1], dim) / norm_dist;

    status = {
        progress,
        generation_progress,
        double(opt.NIndsFront) / double(std::max(1, opt.NIndsFrontMax)),
        opt.SuccessRate,
        opt.EB_hybrid_rate,
        mean_f,
        sigma_f,
        memory_f_mean,
        memory_f_std,
        memory_cr_mean,
        memory_cr_std,
        imitation_slog(opt.FitArrFront[0]),
        imitation_slog(fit_mean),
        std::log1p(std::fabs(fit_std)),
        imitation_slog(opt.FitArrFront[opt.NIndsFront / 2]),
        imitation_slog(opt.FitArrFront[opt.NIndsFront - 1]),
        imitation_slog(parent_fit),
        double(chosen) / double(std::max(1, opt.NIndsFront - 1)),
        double(psizeval) / double(std::max(1, opt.NIndsFront)),
        double(psizeval2) / double(std::max(1, opt.NIndsFront)),
        1.0 - progress,
        std::log1p(double(dim)),
        std::log1p(span),
        div_mean,
        div_std,
        best_to_worst,
    };

    std::vector<const double*> ordered_x;
    std::vector<double> ordered_f;
    imitation_selected_order(opt.Popul[prand], opt.FitArr[prand],
                             opt.PopulFront[rand1], opt.FitArrFront[rand1],
                             opt.Popul[rand2], opt.FitArr[rand2],
                             ordered_x, ordered_f);

    roles.clear();
    role_fitness.clear();
    roles.reserve(size_t(ImitationRoleNames_global_sop.size()) * size_t(dim));
    role_fitness.reserve(ImitationRoleNames_global_sop.size());
    imitation_append_role(roles, role_fitness, opt.PopulFront[chosen], opt.FitArrFront[chosen], dim);
    imitation_append_role(roles, role_fitness, opt.Popul[chosen], opt.FitArr[chosen], dim);
    imitation_append_role(roles, role_fitness, opt.Popul[prand], opt.FitArr[prand], dim);
    imitation_append_role(roles, role_fitness, opt.PopulFront[0], opt.FitArrFront[0], dim);
    imitation_append_role(roles, role_fitness, opt.PopulFront[opt.NIndsFront - 1], opt.FitArrFront[opt.NIndsFront - 1], dim);
    imitation_append_role(roles, role_fitness, opt.PopulFront[opt.NIndsFront / 2], opt.FitArrFront[opt.NIndsFront / 2], dim);
    imitation_append_role(roles, role_fitness, opt.PopulFront[rand1], opt.FitArrFront[rand1], dim);
    imitation_append_role(roles, role_fitness, opt.Popul[rand2], opt.FitArr[rand2], dim);
    imitation_append_role(roles, role_fitness, ordered_x[0], ordered_f[0], dim);
    imitation_append_role(roles, role_fitness, ordered_x[1], ordered_f[1], dim);
    imitation_append_role(roles, role_fitness, ordered_x[2], ordered_f[2], dim);
    imitation_append_role(roles, role_fitness, opt.Popul[opt.Indices[0]], opt.FitArr[opt.Indices[0]], dim);
    imitation_append_role(roles, role_fitness, opt.Popul[opt.Indices[opt.NIndsFront / 2]], opt.FitArr[opt.Indices[opt.NIndsFront / 2]], dim);
    imitation_append_role(roles, role_fitness, opt.Popul[opt.Indices[opt.NIndsFront - 1]], opt.FitArr[opt.Indices[opt.NIndsFront - 1]], dim);
}

static double policy_clamp01(double x, double fallback) {
    if (!std::isfinite(x)) return fallback;
    return std::min(1.0, std::max(0.0, x));
}

static bool maybe_apply_policy_control_callback(Optimizer& opt,
                                                int nfe_before,
                                                int chosen,
                                                int prand,
                                                int rand1,
                                                int rand2,
                                                bool& use_eb,
                                                int will_crossover,
                                                double mean_f,
                                                double sigma_f,
                                                int psizeval,
                                                int psizeval2,
                                                double& f_value,
                                                double& cr_value,
                                                bool& perturbation,
                                                std::vector<char>& policy_mask) {
    if (!PolicyControlCallbackEnabled_global_sop || !PolicyControlCallback_global_sop) {
        return false;
    }

    std::vector<double> status;
    std::vector<double> roles;
    std::vector<double> role_fitness;
    build_imitation_context_vectors(opt,
                                    nfe_before,
                                    chosen,
                                    prand,
                                    rand1,
                                    rand2,
                                    mean_f,
                                    sigma_f,
                                    psizeval,
                                    psizeval2,
                                    status,
                                    roles,
                                    role_fitness);

    const double progress = MaxFEval > 0 ? double(nfe_before) / double(MaxFEval) : 1.0;
    const std::vector<double> rdex_controls = {
        f_value,
        cr_value,
        use_eb ? 1.0 : 0.0,
        perturbation ? 1.0 : 0.0,
        double(will_crossover) / double(std::max(1, opt.NVars - 1)),
        mean_f,
        sigma_f,
        progress,
    };

    const std::vector<double> out =
        PolicyControlCallback_global_sop(status, roles, role_fitness, rdex_controls);
    if (out.size() >= 1) {
        f_value = policy_clamp01(out[0], f_value);
    }
    if (out.size() >= 2) {
        cr_value = policy_clamp01(out[1], cr_value);
    }
    if (out.size() >= 3 && std::isfinite(out[2])) {
        const double p = policy_clamp01(out[2], use_eb ? 1.0 : 0.0);
        use_eb = Random(0, 1) < p;
    }
    if (out.size() >= 4 && std::isfinite(out[3])) {
        const double p = policy_clamp01(out[3], perturbation ? 1.0 : 0.0);
        perturbation = Random(0, 1) < p;
    }

    if (out.size() < size_t(4 + opt.NVars)) {
        return false;
    }

    policy_mask.assign(size_t(opt.NVars), 0);
    int selected = 0;
    for (int j = 0; j < opt.NVars; ++j) {
        const double p = policy_clamp01(out[size_t(4 + j)], 0.0);
        const bool keep = Random(0, 1) < p;
        policy_mask[size_t(j)] = keep ? 1 : 0;
        selected += keep ? 1 : 0;
    }
    if (selected == 0 && opt.NVars > 0) {
        const int forced = std::max(0, std::min(will_crossover, opt.NVars - 1));
        policy_mask[size_t(forced)] = 1;
    }
    return true;
}

static bool maybe_apply_open_action_callback(Optimizer& opt,
                                             int nfe_before,
                                             int chosen,
                                             int prand,
                                             int rand1,
                                             int rand2,
                                             bool use_eb,
                                             bool perturbation,
                                             int will_crossover,
                                             int memory_current_index,
                                             int memory_current_index2,
                                             double rand_eb,
                                             double mean_f,
                                             double sigma_f,
                                             int psizeval,
                                             int psizeval2,
                                             double f_value,
                                             double cr_value,
                                             double normal_f_noise,
                                             int normal_f_rejections,
                                             double normal_cr_noise,
                                             double eb_f_noise,
                                             int eb_f_rejections,
                                             double eb_cr_noise,
                                             double perturbation_uniform,
                                             const std::vector<double>& crossover_uniform,
                                             const std::vector<double>& perturbation_candidate_delta,
                                             const std::vector<double>& repair_candidate_delta,
                                             const std::vector<double>& repair_applied,
                                             const std::vector<double>& perturbation_cauchy_noise,
                                             const std::vector<double>& pre_repair_trial_delta,
                                             const std::vector<double>& pre_repair_trial_x,
                                             double* trial,
                                             std::vector<char>& realized_mask,
                                             double& actual_cr) {
    if (!OpenActionCallbackEnabled_global_sop || !OpenActionCallback_global_sop) {
        return false;
    }

    std::vector<double> status;
    std::vector<double> roles;
    std::vector<double> role_fitness;
    build_imitation_context_vectors(opt,
                                    nfe_before,
                                    chosen,
                                    prand,
                                    rand1,
                                    rand2,
                                    mean_f,
                                    sigma_f,
                                    psizeval,
                                    psizeval2,
                                    status,
                                    roles,
                                    role_fitness);

    std::vector<double> memory_f;
    std::vector<double> memory_cr;
    memory_f.reserve(size_t(opt.MemorySize));
    memory_cr.reserve(size_t(opt.MemorySize));
    for (int i = 0; i < opt.MemorySize; ++i) {
        memory_f.push_back(opt.MemoryF[i]);
        memory_cr.push_back(opt.MemoryCr[i]);
    }

    const double progress = MaxFEval > 0 ? double(nfe_before) / double(MaxFEval) : 1.0;
    const double eb_threshold = opt.EB_hybrid_rate / std::max(1e-12, 1.0 - progress);
    const double eb_f_mu = memory_current_index2 < opt.MemorySize
                               ? opt.MemoryF[memory_current_index2]
                               : F_cauchy_fallback_mu_global_sop;
    const double eb_cr_mu = memory_current_index2 < opt.MemorySize
                                ? opt.MemoryCr[memory_current_index2]
                                : Cr_fallback_mu_global_sop;
    const double normal_cr_mu = memory_current_index < opt.MemorySize
                                    ? opt.MemoryCr[memory_current_index]
                                    : 1.0;
    const std::vector<double> mechanism_meta = {
        double(memory_current_index),
        double(memory_current_index2),
        rand_eb,
        eb_threshold,
        mean_f,
        sigma_f,
        normal_cr_mu,
        0.05,
        eb_f_mu,
        F_cauchy_sigma_global_sop,
        eb_cr_mu,
        Cr_sigma_global_sop,
        double(psizeval),
        double(psizeval2),
        double(will_crossover) / double(std::max(1, opt.NVars - 1)),
        progress,
        normal_f_noise,
        double(normal_f_rejections),
        normal_cr_noise,
        eb_f_noise,
        double(eb_f_rejections),
        eb_cr_noise,
        perturbation_uniform,
    };
    const std::vector<double> rdex_controls = {
        f_value,
        cr_value,
        use_eb ? 1.0 : 0.0,
        perturbation ? 1.0 : 0.0,
        double(will_crossover) / double(std::max(1, opt.NVars - 1)),
        mean_f,
        sigma_f,
        progress,
    };

    const double span = std::max(1e-12, opt.Right - opt.Left);
    const double* target = opt.PopulFront[chosen];
    std::vector<double> rdex_precallback_trial_delta(size_t(opt.NVars), 0.0);
    std::vector<double> rdex_precallback_trial_x(size_t(opt.NVars), 0.0);
    for (int j = 0; j < opt.NVars; ++j) {
        rdex_precallback_trial_delta[size_t(j)] = (trial[j] - target[j]) / span;
        rdex_precallback_trial_x[size_t(j)] = trial[j];
    }

    std::vector<double> vector_materials;
    vector_materials.reserve(size_t(opt.NVars) * size_t(ImitationVectorMaterialNames_global_sop.size()));
    auto append_material = [&](const std::vector<double>& values) {
        for (int j = 0; j < opt.NVars; ++j) {
            vector_materials.push_back(
                values.size() == size_t(opt.NVars)
                    ? values[size_t(j)]
                    : std::numeric_limits<double>::quiet_NaN());
        }
    };
    append_material(crossover_uniform);
    append_material(perturbation_candidate_delta);
    append_material(repair_candidate_delta);
    append_material(repair_applied);
    append_material(rdex_precallback_trial_delta);
    append_material(rdex_precallback_trial_x);
    append_material(perturbation_cauchy_noise);
    append_material(pre_repair_trial_delta);
    append_material(pre_repair_trial_x);

    const std::vector<double> out =
        OpenActionCallback_global_sop(status,
                                      roles,
                                      role_fitness,
                                      memory_f,
                                      memory_cr,
                                      mechanism_meta,
                                      rdex_controls,
                                      vector_materials);
    const int dim = opt.NVars;
    if (out.size() < size_t(dim)) {
        return false;
    }

    realized_mask.assign(size_t(dim), 0);
    std::vector<double> proposed_delta(size_t(dim), 0.0);
    actual_cr = 0.0;
    for (int j = 0; j < dim; ++j) {
        const double delta = std::isfinite(out[size_t(j)]) ? out[size_t(j)] : 0.0;
        proposed_delta[size_t(j)] = delta;
        bool active = std::fabs(delta) > 1.0e-12;
        bool reported_active = active;
        if (out.size() >= size_t(3 * dim)) {
            active = policy_clamp01(out[size_t(dim + j)], active ? 1.0 : 0.0) >= 0.5;
            reported_active = policy_clamp01(out[size_t(2 * dim + j)], active ? 1.0 : 0.0) >= 0.5;
        } else if (out.size() >= size_t(2 * dim)) {
            const double p = policy_clamp01(out[size_t(dim + j)], active ? 1.0 : 0.0);
            active = Random(0, 1) < p;
            reported_active = active;
        }
        const bool has_raw_trial = out.size() >= size_t(4 * dim);
        const double raw_trial =
            has_raw_trial && std::isfinite(out[size_t(3 * dim + j)])
                ? out[size_t(3 * dim + j)]
                : target[j] + span * delta;
        trial[j] = active ? raw_trial : target[j];
        realized_mask[size_t(j)] = reported_active ? 1 : 0;
        actual_cr += reported_active ? 1.0 : 0.0;
    }
    if (actual_cr <= 0.0 && dim > 0) {
        const int forced = std::max(0, std::min(will_crossover, dim - 1));
        realized_mask[size_t(forced)] = 1;
        if (out.size() >= size_t(4 * dim) &&
            std::isfinite(out[size_t(3 * dim + forced)])) {
            trial[forced] = out[size_t(3 * dim + forced)];
        } else {
            trial[forced] = target[forced] + span * proposed_delta[size_t(forced)];
        }
        actual_cr = 1.0;
    }
    return true;
}

static void maybe_log_imitation_example(Optimizer& opt,
                                        int nfe_before,
                                        int chosen,
                                        int prand,
                                        int rand1,
                                        int rand2,
                                        bool use_eb,
                                        bool perturbation,
                                        bool use_trust_region,
                                        double f_value,
                                        double cr_value,
                                        double actual_cr,
                                        int will_crossover,
                                        int memory_current_index,
                                        int memory_current_index2,
                                        double rand_eb,
                                        double mean_f,
                                        double sigma_f,
                                        int psizeval,
                                        int psizeval2,
                                        double normal_f_noise,
                                        int normal_f_rejections,
                                        double normal_cr_noise,
                                        double eb_f_noise,
                                        int eb_f_rejections,
                                        double eb_cr_noise,
                                        double perturbation_uniform,
                                        const std::vector<char>& realized_mask,
                                        const std::vector<double>& crossover_uniform,
                                        const std::vector<double>& perturbation_candidate_delta,
                                        const std::vector<double>& repair_candidate_delta,
                                        const std::vector<double>& repair_applied,
                                        const std::vector<double>& perturbation_cauchy_noise,
                                        const std::vector<double>& pre_repair_trial_delta,
                                        const std::vector<double>& pre_repair_trial_x,
                                        const double* trial,
                                        double trial_fit) {
    if (!ImitationLoggingEnabled_global_sop) return;
    ImitationSeenUpdates_global_sop += 1;
    if (ImitationStride_global_sop > 1 &&
        ((ImitationSeenUpdates_global_sop - 1) % ImitationStride_global_sop) != 0) {
        return;
    }
    const int current_examples =
        int(g_imitation_delta.size() / size_t(std::max(1, opt.NVars)));
    if (ImitationMaxExamples_global_sop > 0 && current_examples >= ImitationMaxExamples_global_sop) {
        return;
    }

    const int dim = opt.NVars;
    const double span = std::max(1e-12, opt.Right - opt.Left);
    const double* target = opt.PopulFront[chosen];
    const double parent_fit = opt.FitArrFront[chosen];

    std::vector<double> status;
    std::vector<double> roles;
    std::vector<double> role_fitness;
    build_imitation_context_vectors(opt,
                                    nfe_before,
                                    chosen,
                                    prand,
                                    rand1,
                                    rand2,
                                    mean_f,
                                    sigma_f,
                                    psizeval,
                                    psizeval2,
                                    status,
                                    roles,
                                    role_fitness);
    g_imitation_status.insert(g_imitation_status.end(), status.begin(), status.end());
    g_imitation_roles.insert(g_imitation_roles.end(), roles.begin(), roles.end());
    g_imitation_role_fitness.insert(g_imitation_role_fitness.end(),
                                    role_fitness.begin(),
                                    role_fitness.end());

    for (int j = 0; j < dim; ++j) {
        g_imitation_mask.push_back(realized_mask.size() == size_t(dim) && realized_mask[size_t(j)] ? 1.0 : 0.0);
        g_imitation_delta.push_back((trial[j] - target[j]) / span);
        g_imitation_trial_x.push_back(trial[j]);
    }

    const double improvement = parent_fit - trial_fit;
    const std::vector<double> action_meta = {
        f_value,
        cr_value,
        actual_cr,
        use_eb ? 1.0 : 0.0,
        perturbation ? 1.0 : 0.0,
        use_trust_region ? 1.0 : 0.0,
        imitation_slog(parent_fit),
        imitation_slog(trial_fit),
        imitation_slog(improvement),
        trial_fit <= parent_fit ? 1.0 : 0.0,
    };
    g_imitation_action_meta.insert(g_imitation_action_meta.end(),
                                   action_meta.begin(),
                                   action_meta.end());

    const std::vector<int> int_meta = {
        opt.func_num,
        opt.Generation,
        nfe_before,
        chosen,
        prand,
        rand1,
        rand2,
        will_crossover,
    };
    g_imitation_int_meta.insert(g_imitation_int_meta.end(), int_meta.begin(), int_meta.end());

    for (int i = 0; i < opt.MemorySize; ++i) {
        g_imitation_memory_f.push_back(opt.MemoryF[i]);
        g_imitation_memory_cr.push_back(opt.MemoryCr[i]);
    }
    const double progress = MaxFEval > 0 ? double(nfe_before) / double(MaxFEval) : 1.0;
    const double eb_threshold = opt.EB_hybrid_rate / std::max(1e-12, 1.0 - progress);
    const double eb_f_mu = memory_current_index2 < opt.MemorySize
                               ? opt.MemoryF[memory_current_index2]
                               : F_cauchy_fallback_mu_global_sop;
    const double eb_cr_mu = memory_current_index2 < opt.MemorySize
                                ? opt.MemoryCr[memory_current_index2]
                                : Cr_fallback_mu_global_sop;
    const double normal_cr_mu = memory_current_index < opt.MemorySize
                                    ? opt.MemoryCr[memory_current_index]
                                    : 1.0;
    const std::vector<double> mechanism_meta = {
        double(memory_current_index),
        double(memory_current_index2),
        rand_eb,
        eb_threshold,
        mean_f,
        sigma_f,
        normal_cr_mu,
        0.05,
        eb_f_mu,
        F_cauchy_sigma_global_sop,
        eb_cr_mu,
        Cr_sigma_global_sop,
        double(psizeval),
        double(psizeval2),
        double(will_crossover) / double(std::max(1, dim - 1)),
        progress,
        normal_f_noise,
        double(normal_f_rejections),
        normal_cr_noise,
        eb_f_noise,
        double(eb_f_rejections),
        eb_cr_noise,
        perturbation_uniform,
    };
    g_imitation_mechanism_meta.insert(g_imitation_mechanism_meta.end(),
                                      mechanism_meta.begin(),
                                      mechanism_meta.end());

    const double nan_value = std::numeric_limits<double>::quiet_NaN();
    auto append_vector_material = [&](const std::vector<double>& values) {
        for (int j = 0; j < dim; ++j) {
            g_imitation_vector_materials.push_back(
                values.size() == size_t(dim) ? values[size_t(j)] : nan_value);
        }
    };
    append_vector_material(crossover_uniform);
    append_vector_material(perturbation_candidate_delta);
    append_vector_material(repair_candidate_delta);
    append_vector_material(repair_applied);
    std::vector<double> rdex_precallback_trial_delta(size_t(dim), 0.0);
    std::vector<double> rdex_precallback_trial_x(size_t(dim), 0.0);
    for (int j = 0; j < dim; ++j) {
        rdex_precallback_trial_delta[size_t(j)] = (trial[j] - target[j]) / span;
        rdex_precallback_trial_x[size_t(j)] = trial[j];
    }
    append_vector_material(rdex_precallback_trial_delta);
    append_vector_material(rdex_precallback_trial_x);
    append_vector_material(perturbation_cauchy_noise);
    append_vector_material(pre_repair_trial_delta);
    append_vector_material(pre_repair_trial_x);
}

void Optimizer::Initialize(int _newNInds, int _newNVars, int _newfunc_num, int _newfunc_index) {
    NVars = _newNVars;
    NIndsCurrent = _newNInds;
    NIndsFront = _newNInds;
    NIndsFrontMax = _newNInds;
    PopulSize = _newNInds * 2;
    Left = Left_platform;
    Right = Right_platform;
    Generation = 0;
    TheChosenOne = 0;
    MemorySize = 5;
    MemoryIter = 0;
    SuccessFilled = 0;
    SuccessRate = 0.5;
    func_num = _newfunc_num;
    func_index = _newfunc_index;
    TrustRegionEnabled = TrustRegionEnabled_global_sop;
    TrustRegionLength = std::max(
        TrustRegionLengthMin_global_sop,
        std::min(TrustRegionLengthInit_global_sop, TrustRegionLengthMax_global_sop)
    );
    TrustRegionSuccessCount = 0;
    TrustRegionFailureCount = 0;
    TrustRegionStagnationCount = 0;
    TrustRegionGenerationActive = false;
    TrustRegionPreviousGenerationActive = false;
    TrustRegionLower.assign(size_t(NVars), Left);
    TrustRegionUpper.assign(size_t(NVars), Right);

    for (int steps_k = 0; steps_k != ResTsize2 - 1; steps_k++)
        stepsFEval[steps_k] = int(10000.0 / double(ResTsize2 - 1) * GNVars * (steps_k + 1));

    Popul = new double*[PopulSize];
    for (int i = 0; i != PopulSize; i++)
        Popul[i] = new double[NVars];
    PopulFront = new double*[NIndsFront];
    for (int i = 0; i != NIndsFront; i++)
        PopulFront[i] = new double[NVars];
    PopulTemp = new double*[PopulSize];
    for (int i = 0; i != PopulSize; i++)
        PopulTemp[i] = new double[NVars];
    FitArr = new double[PopulSize];
    FitArrCopy = new double[PopulSize];
    FitArrFront = new double[NIndsFront];
    Weights = new double[PopulSize];
    tempSuccessCr = new double[PopulSize];
    tempSuccessF = new double[PopulSize];
    FitDelta = new double[PopulSize];
    MemoryCr = new double[MemorySize];
    MemoryF = new double[MemorySize];
    Trial = new double[NVars];
    Indices = new int[PopulSize];
    Indices2 = new int[PopulSize];
    BestInd = new double[NVars];

    for (int i = 0; i < PopulSize; i++)
        for (int j = 0; j < NVars; j++)
            Popul[i][j] = Random(Left, Right);
            
    // Inject custom population if provided
    if (!g_custom_init_pop.empty()) {
        int custom_pop_size = g_custom_init_pop.size() / NVars;
        int max_override = std::min(custom_pop_size, PopulSize);
        for (int i = 0; i < max_override; i++) {
            for (int j = 0; j < NVars; j++) {
                Popul[i][j] = g_custom_init_pop[i * NVars + j];
            }
        }
        // IMPORTANT: Clear it after one use so it doesn't affect future runs
        g_custom_init_pop.clear();
    }
            
    for (int i = 0; i != PopulSize; i++)
        tempSuccessCr[i] = 0;
    for (int i = 0; i != PopulSize; i++)
        tempSuccessF[i] = 0;
    for (int i = 0; i != MemorySize; i++)
        MemoryCr[i] = 1.0;
    for (int i = 0; i != MemorySize; i++)
        MemoryF[i] = 1.0;

    EB_hybrid_flag = new double[NIndsFrontMax];
    FitMass = new double[NIndsFrontMax];
    FitTemp = new double[NIndsFrontMax];
    EB_hybrid_rate = EB_hybrid_rate_init;
}

void Optimizer::UpdateMemoryCr() {
    if (SuccessFilled != 0) {
        MemoryCr[MemoryIter] =
            0.5 * (MeanWL(tempSuccessCr, FitDelta) + MemoryCr[MemoryIter]);
        MemoryF[MemoryIter] = MeanWL(tempSuccessF, FitDelta);
        MemoryIter = (MemoryIter + 1) % MemorySize;
    }
}

double Optimizer::MeanWL(double* Vector, double* TempWeights) {
    double SumWeight = 0;
    double SumSquare = 0;
    double Sum = 0;
    for (int i = 0; i != SuccessFilled; i++)
        SumWeight += TempWeights[i];
    for (int i = 0; i != SuccessFilled; i++)
        Weights[i] = TempWeights[i] / SumWeight;
    for (int i = 0; i != SuccessFilled; i++)
        SumSquare += Weights[i] * Vector[i] * Vector[i];
    for (int i = 0; i != SuccessFilled; i++)
        Sum += Weights[i] * Vector[i];
    if (fabs(Sum) > 1e-8)
        return SumSquare / Sum;
    else
        return 1.0;
}

void Optimizer::FindNSaveBest(bool init, int IndIter) {
    if (FitArr[IndIter] <= bestfit || init)
        bestfit = FitArr[IndIter];
    if (bestfit < globalbest || init) {
        globalbest = bestfit;
        // 记录当前最优个体的决策向量（仅用于输出，不影响算法行为）
        if (Popul && BestInd) {
            for (int j = 0; j < NVars; ++j) {
                BestInd[j] = Popul[IndIter][j];
            }
        }
    }
}

void Optimizer::RemoveWorst(int _NIndsFront, int _newNIndsFront) {
    int PointsToRemove = _NIndsFront - _newNIndsFront;
    for (int L = 0; L != PointsToRemove; L++) {
        double WorstFit = FitArrFront[0];
        int WorstNum = 0;
        for (int i = 1; i != _NIndsFront; i++) {
            if (FitArrFront[i] > WorstFit) {
                WorstFit = FitArrFront[i];
                WorstNum = i;
            }
        }
        for (int i = WorstNum; i != _NIndsFront - 1; i++) {
            for (int j = 0; j != NVars; j++)
                PopulFront[i][j] = PopulFront[i + 1][j];
            FitArrFront[i] = FitArrFront[i + 1];
            FitMass[i] = FitMass[i + 1];
        }
    }
}

void Optimizer::PrepareTrustRegionGeneration() {
    if (!TrustRegionEnabled) {
        TrustRegionGenerationActive = false;
        return;
    }

    const double progress =
        (MaxFEval > 0) ? (double(NFEval) / double(MaxFEval)) : 1.0;
    bool active = false;
    switch (TrustRegionActivationMode_global_sop) {
        case 1:
            active = progress >= TrustRegionLateStartFrac_global_sop;
            break;
        case 2:
            active = TrustRegionStagnationCount >= TrustRegionStagnationTol_global_sop;
            break;
        case 3:
            active = progress >= TrustRegionLateStartFrac_global_sop &&
                     TrustRegionStagnationCount >= TrustRegionStagnationTol_global_sop;
            break;
        case 4:
            if (progress < TrustRegionLateStartFrac_global_sop) {
                TrustRegionStagnationCount = 0;
                active = false;
            } else {
                active = TrustRegionStagnationCount >= TrustRegionStagnationTol_global_sop;
            }
            break;
        default:
            active = true;
            break;
    }

    if (active && !TrustRegionPreviousGenerationActive) {
        TrustRegionLength = std::max(
            TrustRegionLengthMin_global_sop,
            std::min(TrustRegionLengthInit_global_sop, TrustRegionLengthMax_global_sop)
        );
        TrustRegionSuccessCount = 0;
        TrustRegionFailureCount = 0;
    }
    TrustRegionGenerationActive = active;
    if (TrustRegionGenerationActive) {
        RefreshTrustRegionBox();
    }
}

void Optimizer::RefreshTrustRegionBox() {
    if (!TrustRegionEnabled) return;
    const double half = 0.5 * TrustRegionLength;
    for (int j = 0; j < NVars; ++j) {
        double lo = Left;
        double hi = Right;
        if (globalbestinit && BestInd != nullptr) {
            const double center01 = tr_normalize01(BestInd[j], Left, Right);
            const double lo01 = std::max(0.0, center01 - half);
            const double hi01 = std::min(1.0, center01 + half);
            lo = tr_denormalize01(lo01, Left, Right);
            hi = tr_denormalize01(hi01, Left, Right);
        }
        TrustRegionLower[size_t(j)] = lo;
        TrustRegionUpper[size_t(j)] = hi;
    }
}

bool Optimizer::TrialUsesTrustRegion() const {
    if (!TrustRegionEnabled || !TrustRegionGenerationActive) {
        return false;
    }
    if (TrustRegionMixRate_global_sop >= 1.0) {
        return true;
    }
    if (TrustRegionMixRate_global_sop <= 0.0) {
        return false;
    }
    if (TrustRegionDeterministicMix_global_sop) {
        unsigned h = unsigned(Generation + 1) * 2654435761u;
        h ^= unsigned(TheChosenOne + 1) * 2246822519u;
        h ^= unsigned(func_num + 1) * 3266489917u;
        h ^= h >> 16;
        h *= 2246822519u;
        h ^= h >> 13;
        h *= 3266489917u;
        h ^= h >> 16;
        const double u = double(h & 0x00ffffffu) / double(0x01000000u);
        return u < TrustRegionMixRate_global_sop;
    }
    return Random(0.0, 1.0) < TrustRegionMixRate_global_sop;
}

void Optimizer::RepairTrialInTrustRegion() {
    if (!TrustRegionEnabled) return;
    for (int j = 0; j < NVars; ++j) {
        double lo = std::max(Left, TrustRegionLower[size_t(j)]);
        double hi = std::min(Right, TrustRegionUpper[size_t(j)]);
        if (hi < lo) {
            lo = Left;
            hi = Right;
        }
        if (Trial[j] < lo || Trial[j] > hi) {
            Trial[j] = (hi > lo) ? Random(lo, hi) : lo;
        }
    }
}

void Optimizer::UpdateTrustRegionAfterGeneration(double generation_best_before) {
    if (!TrustRegionEnabled) return;
    const bool improved =
        globalbestinit && (!std::isfinite(generation_best_before) || globalbest < generation_best_before);
    if (improved) {
        TrustRegionStagnationCount = 0;
    } else {
        TrustRegionStagnationCount += 1;
    }

    if (TrustRegionGenerationActive) {
        if (improved) {
            TrustRegionSuccessCount += 1;
            TrustRegionFailureCount = 0;
            if (TrustRegionSuccessCount >= std::max(1, TrustRegionSuccessTol_global_sop)) {
                TrustRegionLength = std::min(TrustRegionLength * 2.0, TrustRegionLengthMax_global_sop);
                TrustRegionSuccessCount = 0;
            }
        } else {
            TrustRegionFailureCount += 1;
            TrustRegionSuccessCount = 0;
            if (TrustRegionFailureCount >= std::max(1, TrustRegionFailureTol_global_sop)) {
                TrustRegionLength *= 0.5;
                TrustRegionFailureCount = 0;
                if (TrustRegionLength < TrustRegionLengthMin_global_sop) {
                    TrustRegionLength = TrustRegionLengthInit_global_sop;
                    TrustRegionSuccessCount = 0;
                    TrustRegionFailureCount = 0;
                }
            }
        }
    }
    TrustRegionPreviousGenerationActive = TrustRegionGenerationActive;
}

void Optimizer::EB_order(int prand, int Rand1, int Rand2) {
    double* pos1 = Popul[prand];
    double pos1Fit = FitArr[prand];
    double* pos3 = PopulFront[Rand1];
    double pos3Fit = FitArrFront[Rand1];
    double* pos4_arch = Popul[Rand2];
    double pos4Fit_arch = FitArr[Rand2];
    if (pos1Fit <= pos3Fit && pos1Fit <= pos4Fit_arch) {
        ord_best_arch = pos1;
        if (pos3Fit <= pos4Fit_arch) {
            ord_medium_arch = pos3;
            ord_worst_arch = pos4_arch;
        } else {
            ord_medium_arch = pos4_arch;
            ord_worst_arch = pos3;
        }
    } else if (pos3Fit <= pos1Fit && pos3Fit <= pos4Fit_arch) {
        ord_best_arch = pos3;
        if (pos1Fit <= pos4Fit_arch) {
            ord_medium_arch = pos1;
            ord_worst_arch = pos4_arch;
        } else {
            ord_medium_arch = pos4_arch;
            ord_worst_arch = pos1;
        }
    } else if (pos4Fit_arch <= pos1Fit && pos4Fit_arch <= pos3Fit) {
        ord_best_arch = pos4_arch;
        if (pos1Fit <= pos3Fit) {
            ord_medium_arch = pos1;
            ord_worst_arch = pos3;
        } else {
            ord_medium_arch = pos3;
            ord_worst_arch = pos1;
        }
    }
    double* pos4_popul = Popul[Rand2];
    double pos4Fit_popul = FitArr[Rand2];
    if (pos1Fit <= pos3Fit && pos1Fit <= pos4Fit_popul) {
        ord_best_popul = pos1;
        if (pos3Fit <= pos4Fit_popul) {
            ord_medium_popul = pos3;
            ord_worst_popul = pos4_popul;
        } else {
            ord_medium_popul = pos4_popul;
            ord_worst_popul = pos3;
        }
    } else if (pos3Fit <= pos1Fit && pos3Fit <= pos4Fit_popul) {
        ord_best_popul = pos3;
        if (pos1Fit <= pos4Fit_popul) {
            ord_medium_popul = pos1;
            ord_worst_popul = pos4_popul;
        } else {
            ord_medium_popul = pos4_popul;
            ord_worst_popul = pos1;
        }
    } else if (pos4Fit_popul <= pos1Fit && pos4Fit_popul <= pos3Fit) {
        ord_best_popul = pos4_popul;
        if (pos1Fit <= pos3Fit) {
            ord_medium_popul = pos1;
            ord_worst_popul = pos3;
        } else {
            ord_medium_popul = pos3;
            ord_worst_popul = pos1;
        }
    }
}

void Optimizer::UpdateEB_hybrid_param(double* EB_hybrid_flag_local,
                                      double* FitArrFront_local,
                                      vector<double> FitTemp_local) {
    double SumEB_DeltaFit = 0;
    double SumOrigin_DeltaFit = 0;
    for (int ChosenOne = 0; ChosenOne != NIndsFront; ChosenOne++) {
        if (EB_hybrid_flag_local[ChosenOne] == 1) {
            if (FitTemp_local[ChosenOne] <= FitArrFront_local[ChosenOne]) {
                SumEB_DeltaFit += FitArrFront_local[ChosenOne] - FitTemp_local[ChosenOne];
            }
        } else {
            if (FitTemp_local[ChosenOne] <= FitArrFront_local[ChosenOne]) {
                SumOrigin_DeltaFit += FitArrFront_local[ChosenOne] - FitTemp_local[ChosenOne];
            }
        }
    }

    if (SumEB_DeltaFit != 0 && SumOrigin_DeltaFit != 0) {
        EB_hybrid_rate = SumEB_DeltaFit / (SumEB_DeltaFit + SumOrigin_DeltaFit);
        double EB_limit_min = 0;
        double EB_limit_max = 1;
        if (EB_hybrid_rate > EB_limit_max) {
            EB_hybrid_rate = EB_limit_max;
        } else if (EB_hybrid_rate < EB_limit_min) {
            EB_hybrid_rate = EB_limit_min;
        }
    } else {
        EB_hybrid_rate = EB_hybrid_rate_init;
    }
}

static std::vector<double> rbf_current_best_normalized_point(Optimizer& opt) {
    int best_index = 0;
    for (int i = 1; i < opt.NIndsCurrent; ++i) {
        if (opt.FitArr[i] < opt.FitArr[best_index]) {
            best_index = i;
        }
    }
    return rbf_normalize_point(opt.Popul[best_index], opt.NVars);
}

static std::vector<int> rbf_select_indices_reference(Optimizer& opt, int max_samples) {
    std::vector<int> selected;
    const int archive_count = rbf_archive_count(opt.NVars);
    if (archive_count <= max_samples) {
        selected.resize(size_t(archive_count));
        std::iota(selected.begin(), selected.end(), 0);
        return selected;
    }

    std::vector<double> center = rbf_current_best_normalized_point(opt);
    std::vector<int> indices(size_t(archive_count), 0);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int lhs, int rhs) {
        return rbf_distance_normalized_to_archive(center, lhs, opt.NVars) <
               rbf_distance_normalized_to_archive(center, rhs, opt.NVars);
    });

    const int local_target =
        std::min(max_samples, std::max(2 * opt.NVars, int(max_samples * 0.7)));
    const int local_pool =
        std::min(archive_count, std::max(RbfLocalPoolCap_global_sop, max_samples));
    std::vector<bool> used(size_t(archive_count), false);
    for (int i = 0; i < local_target; ++i) {
        selected.push_back(indices[size_t(i)]);
        used[size_t(indices[size_t(i)])] = true;
    }

    while (int(selected.size()) < max_samples) {
        int best_candidate = -1;
        double best_score = -1.0;
        for (int pool_pos = 0; pool_pos < local_pool; ++pool_pos) {
            const int candidate = indices[size_t(pool_pos)];
            if (used[size_t(candidate)]) continue;
            double min_distance = std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < selected.size(); ++j) {
                const double distance =
                    rbf_distance_archive_archive(candidate, selected[j], opt.NVars);
                min_distance = std::min(min_distance, distance);
            }
            if (min_distance > best_score) {
                best_score = min_distance;
                best_candidate = candidate;
            }
        }
        if (best_candidate < 0) break;
        selected.push_back(best_candidate);
        used[size_t(best_candidate)] = true;
    }

    for (int i = 0; int(selected.size()) < max_samples && i < archive_count; ++i) {
        const int candidate = indices[size_t(i)];
        if (used[size_t(candidate)]) continue;
        selected.push_back(candidate);
    }
    return selected;
}

static std::vector<int> rbf_select_indices_fast(Optimizer& opt, int max_samples) {
    std::vector<int> selected;
    const int archive_count = rbf_archive_count(opt.NVars);
    if (archive_count <= max_samples) {
        selected.resize(size_t(archive_count));
        std::iota(selected.begin(), selected.end(), 0);
        return selected;
    }

    std::vector<double> center = rbf_current_best_normalized_point(opt);
    std::vector<double> center_dist(size_t(archive_count), 0.0);
    std::vector<int> indices(size_t(archive_count), 0);
    for (int i = 0; i < archive_count; ++i) {
        indices[size_t(i)] = i;
        center_dist[size_t(i)] = rbf_distance_normalized_to_archive(center, i, opt.NVars);
    }
    std::sort(indices.begin(), indices.end(), [&](int lhs, int rhs) {
        return center_dist[size_t(lhs)] < center_dist[size_t(rhs)];
    });

    const int local_target =
        std::min(max_samples, std::max(2 * opt.NVars, int(max_samples * 0.7)));
    const int local_pool =
        std::min(archive_count, std::max(RbfLocalPoolCap_global_sop, max_samples));
    std::vector<bool> used(size_t(archive_count), false);
    for (int i = 0; i < local_target; ++i) {
        const int idx = indices[size_t(i)];
        selected.push_back(idx);
        used[size_t(idx)] = true;
    }

    std::vector<double> nearest(size_t(archive_count), std::numeric_limits<double>::infinity());
    for (int pool_pos = 0; pool_pos < local_pool; ++pool_pos) {
        const int candidate = indices[size_t(pool_pos)];
        if (used[size_t(candidate)]) continue;
        double min_distance = std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < selected.size(); ++j) {
            const double distance =
                rbf_distance_archive_archive(candidate, selected[j], opt.NVars);
            min_distance = std::min(min_distance, distance);
        }
        nearest[size_t(candidate)] = min_distance;
    }

    while (int(selected.size()) < max_samples) {
        int best_candidate = -1;
        double best_score = -1.0;
        for (int pool_pos = 0; pool_pos < local_pool; ++pool_pos) {
            const int candidate = indices[size_t(pool_pos)];
            if (used[size_t(candidate)]) continue;
            const double min_distance = nearest[size_t(candidate)];
            if (min_distance > best_score) {
                best_score = min_distance;
                best_candidate = candidate;
            }
        }
        if (best_candidate < 0) break;
        selected.push_back(best_candidate);
        used[size_t(best_candidate)] = true;
        for (int pool_pos = 0; pool_pos < local_pool; ++pool_pos) {
            const int candidate = indices[size_t(pool_pos)];
            if (used[size_t(candidate)]) continue;
            const double distance =
                rbf_distance_archive_archive(candidate, best_candidate, opt.NVars);
            nearest[size_t(candidate)] = std::min(nearest[size_t(candidate)], distance);
        }
    }

    for (int i = 0; int(selected.size()) < max_samples && i < archive_count; ++i) {
        const int candidate = indices[size_t(i)];
        if (used[size_t(candidate)]) continue;
        selected.push_back(candidate);
    }
    return selected;
}

static bool rbf_solve_linear_system(std::vector<double>& matrix,
                                    std::vector<double>& rhs,
                                    int n) {
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double pivot_abs = std::fabs(matrix[size_t(col) * size_t(n) + size_t(col)]);
        for (int row = col + 1; row < n; ++row) {
            const double candidate =
                std::fabs(matrix[size_t(row) * size_t(n) + size_t(col)]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if (pivot_abs < 1e-14) return false;
        if (pivot != col) {
            for (int k = col; k < n; ++k) {
                std::swap(matrix[size_t(pivot) * size_t(n) + size_t(k)],
                          matrix[size_t(col) * size_t(n) + size_t(k)]);
            }
            std::swap(rhs[size_t(pivot)], rhs[size_t(col)]);
        }

        const double diag = matrix[size_t(col) * size_t(n) + size_t(col)];
        for (int row = col + 1; row < n; ++row) {
            const double factor = matrix[size_t(row) * size_t(n) + size_t(col)] / diag;
            if (std::fabs(factor) < 1e-18) continue;
            for (int k = col; k < n; ++k) {
                matrix[size_t(row) * size_t(n) + size_t(k)] -=
                    factor * matrix[size_t(col) * size_t(n) + size_t(k)];
            }
            rhs[size_t(row)] -= factor * rhs[size_t(col)];
        }
    }

    for (int row = n - 1; row >= 0; --row) {
        double sum = rhs[size_t(row)];
        for (int col = row + 1; col < n; ++col) {
            sum -= matrix[size_t(row) * size_t(n) + size_t(col)] * rhs[size_t(col)];
        }
        rhs[size_t(row)] = sum / matrix[size_t(row) * size_t(n) + size_t(row)];
    }
    return true;
}

static RBFModelSOP rbf_build_local_model(Optimizer& opt) {
    int max_samples = RbfSampleCap_global_sop;
    if (max_samples < opt.NVars + 2) max_samples = opt.NVars + 2;
    std::vector<int> chosen = RbfFastSubsetSelection_global_sop
        ? rbf_select_indices_fast(opt, max_samples)
        : rbf_select_indices_reference(opt, max_samples);
    double selected_signature = 0.0;
    double selected_signature2 = 0.0;
    for (size_t i = 0; i < chosen.size(); ++i) {
        selected_signature += double(i + 1) * double(chosen[i] + 1);
        selected_signature2 += double((i + 1) * (i + 1)) * double(chosen[i] + 1);
    }
    if (LastRBFDiagnostics_global_sop.size() < 8) {
        LastRBFDiagnostics_global_sop.assign(8, 0.0);
    }
    LastRBFDiagnostics_global_sop[6] = selected_signature;
    LastRBFDiagnostics_global_sop[7] = selected_signature2;

    RBFModelSOP model;
    model.dim = opt.NVars;
    model.sample_count = int(chosen.size());
    if (model.sample_count <= model.dim + 1) {
        throw std::runtime_error("Insufficient archive samples for RBF.");
    }
    model.samples.assign(size_t(model.sample_count) * size_t(model.dim), 0.0);

    std::vector<double> fitness(size_t(model.sample_count), 0.0);
    for (int i = 0; i < model.sample_count; ++i) {
        const double* point = rbf_archive_point_ptr(chosen[size_t(i)], opt.NVars);
        for (int j = 0; j < opt.NVars; ++j) {
            model.samples[size_t(i) * size_t(opt.NVars) + size_t(j)] = point[j];
        }
        fitness[size_t(i)] = RbfArchiveFitness_global_sop[size_t(chosen[size_t(i)])];
    }

    model.y_mean = std::accumulate(fitness.begin(), fitness.end(), 0.0) / fitness.size();
    double variance = 0.0;
    for (double y : fitness) {
        const double centered = y - model.y_mean;
        variance += centered * centered;
    }
    model.y_scale = std::sqrt(variance / std::max<size_t>(1, fitness.size()));
    if (model.y_scale < 1e-12) model.y_scale = 1.0;

    std::vector<double> normalized_y(fitness.size(), 0.0);
    for (size_t i = 0; i < fitness.size(); ++i) {
        normalized_y[i] = (fitness[i] - model.y_mean) / model.y_scale;
    }

    const int system_size = model.sample_count + model.dim + 1;
    const double nuggets[] = {1e-10, 1e-8, 1e-6, 1e-4};
    std::vector<double> params;
    bool solved = false;
    for (double nugget : nuggets) {
        std::vector<double> system(size_t(system_size) * size_t(system_size), 0.0);
        std::vector<double> rhs(size_t(system_size), 0.0);
        for (int i = 0; i < model.sample_count; ++i) {
            for (int j = 0; j < model.sample_count; ++j) {
                system[size_t(i) * size_t(system_size) + size_t(j)] =
                    rbf_kernel_cubic_sample_sample(model, i, j);
                if (i == j) {
                    system[size_t(i) * size_t(system_size) + size_t(j)] += nugget;
                }
            }
            for (int j = 0; j < model.dim; ++j) {
                const double v = model.samples[size_t(i) * size_t(model.dim) + size_t(j)];
                system[size_t(i) * size_t(system_size) + size_t(model.sample_count + j)] = v;
                system[size_t(model.sample_count + j) * size_t(system_size) + size_t(i)] = v;
            }
            system[size_t(i) * size_t(system_size) + size_t(system_size - 1)] = 1.0;
            system[size_t(system_size - 1) * size_t(system_size) + size_t(i)] = 1.0;
            rhs[size_t(i)] = normalized_y[size_t(i)];
        }

        if (rbf_solve_linear_system(system, rhs, system_size)) {
            params.swap(rhs);
            solved = true;
            break;
        }
    }
    if (!solved) throw std::runtime_error("RBF solve failed after nugget retries.");

    model.lambda.assign(params.begin(), params.begin() + model.sample_count);
    model.gamma.assign(params.begin() + model.sample_count, params.end());
    return model;
}

static bool rbf_surrogate_early_stop_hook(Optimizer* /*opt*/) {
    if (globalbest + 1e-12 < RbfSurrogateBestSeen_global_sop) {
        RbfSurrogateBestSeen_global_sop = globalbest;
        RbfNoImproveGenerations_global_sop = 0;
    } else {
        RbfNoImproveGenerations_global_sop++;
    }
    return RbfNoImproveGenerations_global_sop >= RbfSurrogatePatience_global_sop;
}

static void rbf_rebuild_front(Optimizer& opt) {
    double minfit = opt.FitArr[0];
    double maxfit = opt.FitArr[0];
    for (int i = 0; i < opt.NIndsFront; ++i) {
        opt.FitArrCopy[i] = opt.FitArr[i];
        opt.Indices[i] = i;
        maxfit = std::max(maxfit, opt.FitArr[i]);
        minfit = std::min(minfit, opt.FitArr[i]);
    }
    if (minfit != maxfit) {
        qSort2int(opt.FitArrCopy, opt.Indices, 0, opt.NIndsFront - 1);
    }
    for (int i = 0; i < opt.NIndsFront; ++i) {
        for (int j = 0; j < opt.NVars; ++j) {
            opt.PopulFront[i][j] = opt.Popul[opt.Indices[i]][j];
        }
        opt.FitArrFront[i] = opt.FitArrCopy[i];
        opt.FitMass[i] = opt.FitArrFront[i];
    }
}

static std::vector<double> rbf_search_surrogate_best(Optimizer& initial_opt,
                                                     RBFModelSOP& model) {
    const int saved_nfe = NFEval;
    const int saved_max_fe = MaxFEval;
    const int saved_last_fe = LastFEcount;
    const double saved_globalbest = globalbest;
    const bool saved_globalbestinit = globalbestinit;
    const bool saved_use_cec = UseCec2017Evaluator_global_sop;
    const bool saved_initial_enabled = RbfInitialInjectionEnabled_global_sop;
    const bool saved_rbf_surrogate = RbfSurrogateEvalMode_global_sop;
    const bool saved_rbf_record = RbfRecordTrueSamples_global_sop;
    RBFModelSOP* saved_active_model = ActiveRBFModel_global_sop;
    bool (*saved_hook)(Optimizer*) = GenerationHook_global_sop;
    double saved_results[ResTsize2];
    std::memcpy(saved_results, ResultsArray, sizeof(ResultsArray));

    ActiveRBFModel_global_sop = &model;
    RbfInitialInjectionEnabled_global_sop = false;
    RbfSurrogateEvalMode_global_sop = true;
    RbfRecordTrueSamples_global_sop = false;
    GenerationHook_global_sop = rbf_surrogate_early_stop_hook;
    RbfSurrogateBestSeen_global_sop = std::numeric_limits<double>::infinity();
    RbfNoImproveGenerations_global_sop = 0;

    NFEval = 0;
    LastFEcount = 0;
    globalbest = std::numeric_limits<double>::infinity();
    globalbestinit = false;
    for (int i = 0; i < ResTsize2; ++i) {
        ResultsArray[i] = double(saved_max_fe);
    }

    Optimizer surrogate_opt;
    surrogate_opt.Initialize(initial_opt.NIndsFront,
                             initial_opt.NVars,
                             initial_opt.func_num,
                             initial_opt.func_index);
    for (int i = 0; i < initial_opt.NIndsFront; ++i) {
        for (int j = 0; j < initial_opt.NVars; ++j) {
            surrogate_opt.Popul[i][j] = initial_opt.Popul[i][j];
        }
    }
    surrogate_opt.MainCycle();

    int best_index = 0;
    double best_fit = rbf_evaluate_point(surrogate_opt.Popul[0], model);
    for (int i = 1; i < surrogate_opt.NIndsCurrent; ++i) {
        const double fit = rbf_evaluate_point(surrogate_opt.Popul[i], model);
        if (fit < best_fit) {
            best_fit = fit;
            best_index = i;
        }
    }
    std::vector<double> best_point(size_t(surrogate_opt.NVars), 0.0);
    for (int j = 0; j < surrogate_opt.NVars; ++j) {
        best_point[size_t(j)] = surrogate_opt.Popul[best_index][j];
    }
    surrogate_opt.Clean();

    ActiveRBFModel_global_sop = saved_active_model;
    UseCec2017Evaluator_global_sop = saved_use_cec;
    RbfInitialInjectionEnabled_global_sop = saved_initial_enabled;
    RbfSurrogateEvalMode_global_sop = saved_rbf_surrogate;
    RbfRecordTrueSamples_global_sop = saved_rbf_record;
    GenerationHook_global_sop = saved_hook;
    NFEval = saved_nfe;
    MaxFEval = saved_max_fe;
    LastFEcount = saved_last_fe;
    globalbest = saved_globalbest;
    globalbestinit = saved_globalbestinit;
    std::memcpy(ResultsArray, saved_results, sizeof(ResultsArray));
    return best_point;
}

static void rbf_inject_surrogate_sample(Optimizer& opt) {
    LastRBFDiagnostics_global_sop.assign(8, 0.0);
    LastRBFDiagnostics_global_sop[0] = 1.0;
    LastRBFDiagnostics_global_sop[3] = double(rbf_archive_count(opt.NVars));
    if (NFEval >= MaxFEval) return;

    RBFModelSOP model;
    std::vector<double> surrogate_best;
    try {
        model = rbf_build_local_model(opt);
        LastRBFDiagnostics_global_sop[2] = double(model.sample_count);
        surrogate_best = rbf_search_surrogate_best(opt, model);
    } catch (const std::runtime_error&) {
        return;
    }

    const double candidate_fit = cec_24(surrogate_best.data(), opt.func_num);
    int worst_index = 0;
    for (int i = 1; i < opt.NIndsFront; ++i) {
        if (opt.FitArr[i] > opt.FitArr[worst_index]) {
            worst_index = i;
        }
    }
    LastRBFDiagnostics_global_sop[4] = candidate_fit;
    LastRBFDiagnostics_global_sop[5] = opt.FitArr[worst_index];

    if (candidate_fit < opt.FitArr[worst_index]) {
        for (int j = 0; j < opt.NVars; ++j) {
            opt.Popul[worst_index][j] = surrogate_best[size_t(j)];
        }
        opt.FitArr[worst_index] = candidate_fit;
        if (candidate_fit < opt.bestfit) {
            opt.bestfit = candidate_fit;
        }
        if (!globalbestinit || candidate_fit < globalbest) {
            globalbest = candidate_fit;
            globalbestinit = true;
            if (opt.BestInd) {
                for (int j = 0; j < opt.NVars; ++j) {
                    opt.BestInd[j] = surrogate_best[size_t(j)];
                }
            }
        }
        LastRBFDiagnostics_global_sop[1] = 1.0;
        rbf_rebuild_front(opt);
    }
    SaveBestValues(opt.func_index);
}

void Optimizer::MainCycle() {
    vector<double> FitTemp2;
    vector<double> FitTemp_prand;
    for (int IndIter = 0; IndIter < NIndsFront; IndIter++) {
        FitArr[IndIter] = cec_24(Popul[IndIter], func_num);
        if (XaiInternalEnabled_global_sop) {
            xai_push_sample(Popul[IndIter], FitArr[IndIter], NVars);
            xai_maybe_update_weights(NVars);
        }
        FindNSaveBest(IndIter == 0, IndIter);
        if (!globalbestinit || bestfit < globalbest) {
            globalbest = bestfit;
            globalbestinit = true;
        }
        SaveBestValues(func_index);
    }
    double minfit = FitArr[0];
    double maxfit = FitArr[0];
    for (int i = 0; i != NIndsFront; i++) {
        FitArrCopy[i] = FitArr[i];
        Indices[i] = i;
        maxfit = max(maxfit, FitArr[i]);
        minfit = min(minfit, FitArr[i]);
    }
    if (minfit != maxfit)
        qSort2int(FitArrCopy, Indices, 0, NIndsFront - 1);
    for (int i = 0; i != NIndsFront; i++) {
        for (int j = 0; j != NVars; j++)
            PopulFront[i][j] = Popul[Indices[i]][j];
        FitArrFront[i] = FitArrCopy[i];
        FitMass[i] = FitArrFront[i];
    }
    if (RbfInitialInjectionEnabled_global_sop) {
        rbf_inject_surrogate_sample(*this);
    }
    PFIndex = 0;
    while (NFEval < MaxFEval) {
        const double generation_best_before =
            globalbestinit ? globalbest : std::numeric_limits<double>::infinity();
        PrepareTrustRegionGeneration();
        double meanF = 0.4 + tanh(SuccessRate * 5) * 0.25;
        double sigmaF = F_sigma_global_sop;
        minfit = FitArr[0];
        maxfit = FitArr[0];
        for (int i = 0; i != NIndsFront; i++) {
            FitArrCopy[i] = FitArr[i];
            Indices[i] = i;
            maxfit = max(maxfit, FitArr[i]);
            minfit = min(minfit, FitArr[i]);
        }
        if (minfit != maxfit)
            qSort2int(FitArrCopy, Indices, 0, NIndsFront - 1);
        minfit = FitArrFront[0];
        maxfit = FitArrFront[0];
        for (int i = 0; i != NIndsFront; i++) {
            FitArrCopy[i] = FitArrFront[i];
            Indices2[i] = i;
            maxfit = max(maxfit, FitArrFront[i]);
            minfit = min(minfit, FitArrFront[i]);
        }
        if (minfit != maxfit)
            qSort2int(FitArrCopy, Indices2, 0, NIndsFront - 1);
        FitTemp2.resize(NIndsFront);
        for (int i = 0; i != NIndsFront; i++)
            // 保持与原始 RDEx_SOP 一致：exp(-i/NIndsFront * 3)
            // 这里暂不参数化以确保默认行为完全等价。
            FitTemp2[i] = exp(-double(i) / double(NIndsFront) * 3.0);
        std::discrete_distribution<int> ComponentSelectorFront(FitTemp2.begin(), FitTemp2.end());

        int prand = 0;
        int Rand1 = 0;
        int Rand2 = 0;
        int psizeval = max(
            2,
            int(NIndsFront * PsizeCoef_global_sop * exp(-SuccessRate * SelectExp_global_sop))
        );
        FitTemp_prand.resize(NIndsFront);
        for (int i = 0; i != NIndsFront; i++)
            FitTemp_prand[i] = 3.0 * (NIndsFront - i);
        int psizeval2 =
            int(NIndsFront * 0.17 * (1 - 0.5 * (double)NFEval / (double)MaxFEval));
        if (psizeval2 <= 1)
            psizeval2 = 2;
        std::discrete_distribution<int> ComponentSelectorFront2(
            FitTemp_prand.begin(), FitTemp_prand.begin() + psizeval2);
        std::discrete_distribution<int> ComponentSelectorFront3(FitTemp_prand.begin(),
                                                                FitTemp_prand.end());

        for (int IndIter = 0; IndIter < NIndsFront; IndIter++) {
            TheChosenOne = IntRandom(NIndsFront);
            MemoryCurrentIndex = IntRandom(MemorySize);
            MemoryCurrentIndex2 = IntRandom(MemorySize + 1);
            do
                prand = Indices[IntRandom(psizeval)];
            while (prand == TheChosenOne);
            do
                Rand1 = Indices2[ComponentSelectorFront(generator_uni_i_3)];
            while (Rand1 == prand);
            do
                Rand2 = Indices[IntRandom(NIndsFront)];
            while (Rand2 == prand || Rand2 == Rand1);
            const double material_nan = std::numeric_limits<double>::quiet_NaN();
            double normalFNoise = material_nan;
            int normalFRejections = 0;
            double normalCrNoise = material_nan;
            double ebFNoise = material_nan;
            int ebFRejections = 0;
            double ebCrNoise = material_nan;
            double perturbationUniform = material_nan;
            do {
                F = NormRandWithNoise(meanF, sigmaF, normalFNoise);
                if (F < 0.0 || F > 1.0) normalFRejections++;
            } while (F < 0.0 || F > 1.0);
            Cr = NormRandWithNoise(MemoryCr[MemoryCurrentIndex], 0.05, normalCrNoise);
            Cr = min(max(Cr, 0.0), 1.0);
            double ActualCr = 0;
            int WillCrossover = IntRandom(NVars);
            const bool use_tr = TrialUsesTrustRegion();

            double Rand_EB = Random(0, 1);
            if ((double)NFEval / (double)MaxFEval < EB_early_disable_frac_global_sop) {
                Rand_EB = EB_rand_multiplier_global_sop;
            }
            bool use_eb = (Rand_EB * (1 - double(NFEval) / double(MaxFEval))) < EB_hybrid_rate;
            bool perturbation = false;
            std::vector<char> realizedMask(size_t(NVars), 0);
            const double material_span = std::max(1e-12, Right - Left);
            std::vector<double> crossoverUniform(size_t(NVars), material_nan);
            std::vector<double> perturbationCandidateDelta(size_t(NVars), material_nan);
            std::vector<double> repairCandidateDelta(size_t(NVars), material_nan);
            std::vector<double> repairApplied(size_t(NVars), 0.0);
            std::vector<double> perturbationCauchyNoise(size_t(NVars), material_nan);
            std::vector<double> preRepairTrialDelta(size_t(NVars), 0.0);
            std::vector<double> preRepairTrialX(size_t(NVars), 0.0);
            if (use_eb) {
                do
                    prand = Indices[ComponentSelectorFront2(generator_uni_i_3)];
                while (prand == TheChosenOne);
                do
                    Rand1 = Indices2[ComponentSelectorFront3(generator_uni_i_3)];
                while (Rand1 == prand);
                do
                    Rand2 = Indices[ComponentSelectorFront3(generator_uni_i_3)];
                while (Rand2 == prand || Rand2 == Rand1);
                EB_order(prand, Rand1, Rand2);
                do {
                    if (MemoryCurrentIndex2 < MemorySize)
                        F = CachyRandWithNoise(MemoryF[MemoryCurrentIndex2], F_cauchy_sigma_global_sop, ebFNoise);
                    else
                        F = CachyRandWithNoise(F_cauchy_fallback_mu_global_sop, F_cauchy_sigma_global_sop, ebFNoise);
                    if (F < 0.0) ebFRejections++;
                } while (F < 0.0);
                if (F > 1.0)
                    F = 1.0;

                if ((double)NFEval / (double)MaxFEval < 0.6 && F > 0.7)
                    F = 0.7;

                if (MemoryCurrentIndex2 < MemorySize) {
                    if (MemoryCr[MemoryCurrentIndex2] < 0)
                        Cr = 0;
                    else
                        Cr = NormRandWithNoise(MemoryCr[MemoryCurrentIndex2], Cr_sigma_global_sop, ebCrNoise);
                } else
                    Cr = NormRandWithNoise(Cr_fallback_mu_global_sop, Cr_sigma_global_sop, ebCrNoise);
                if (Cr >= 1)
                    Cr = 1;
                if (Cr <= 0)
                    Cr = 0;

                if ((double)NFEval / (double)MaxFEval < 0.25)
                    Cr = max(Cr, Cr_early_min1_global_sop);
                if ((double)NFEval / (double)MaxFEval < 0.5)
                    Cr = max(Cr, Cr_early_min2_global_sop);

                std::vector<char> crossoverMask;
                if (WeightedCrossoverEnabled_global_sop) {
                    int target = int(std::round(Cr * double(NVars)));
                    target = std::max(1, std::min(target, NVars));

                    std::vector<double> weights = DimensionWeights_global_sop;
                    if ((int)weights.size() != NVars) {
                        weights.assign(NVars, 1.0);
                    }
                    double sumw = 0.0;
                    for (int j = 0; j < NVars; ++j) {
                        double w = weights[j];
                        if (!std::isfinite(w) || w < 0.0) w = 0.0;
                        weights[j] = w;
                        sumw += w;
                    }
                    if (!(sumw > 0.0)) {
                        std::fill(weights.begin(), weights.end(), 1.0);
                    }

                    std::discrete_distribution<int> dimSelector(weights.begin(), weights.end());
                    crossoverMask.assign(NVars, 0);
                    crossoverMask[WillCrossover] = 1;
                    int filled = 1;
                    while (filled < target) {
                        int idx = dimSelector(generator_uni_i_3);
                        if (!crossoverMask[idx]) {
                            crossoverMask[idx] = 1;
                            filled++;
                        }
                    }
                }

                perturbation = LocalPerturbationEnabledWithUniform(perturbationUniform);
                std::vector<char> policyMask;
                const bool hasPolicyMask = maybe_apply_policy_control_callback(*this,
                                                                                NFEval,
                                                                                TheChosenOne,
                                                                                prand,
                                                                                Rand1,
                                                                                Rand2,
                                                                                use_eb,
                                                                                WillCrossover,
                                                                                meanF,
                                                                                sigmaF,
                                                                                psizeval,
                                                                                psizeval2,
                                                                                F,
                                                                                Cr,
                                                                                perturbation,
                                                                                policyMask);
                EB_hybrid_flag[TheChosenOne] = use_eb ? 1 : 0;

                for (int j = 0; j != NVars; j++) {
                    bool doCross = false;
                    if (hasPolicyMask) {
                        doCross = policyMask[size_t(j)] != 0;
                    } else if (WeightedCrossoverEnabled_global_sop) {
                        doCross = crossoverMask[j] != 0;
                    } else {
                        const double cross_u = Random(0, 1);
                        crossoverUniform[size_t(j)] = cross_u;
                        doCross = cross_u < Cr || WillCrossover == j;
                    }
                    realizedMask[size_t(j)] = doCross ? 1 : 0;
                    if (doCross) {
                        if (use_eb) {
                            Trial[j] = EbDonorValue(PopulFront[TheChosenOne][j],
                                                    ord_best_arch[j],
                                                    ord_medium_arch[j],
                                                    ord_worst_arch[j],
                                                    PopulFront[Rand1][j],
                                                    Popul[Rand2][j],
                                                    F);
                        } else {
                            Trial[j] = PopulFront[TheChosenOne][j] +
                                       F * (Popul[prand][j] - PopulFront[TheChosenOne][j]) +
                                       F * (PopulFront[Rand1][j] - Popul[Rand2][j]);
                        }
                        preRepairTrialX[size_t(j)] = Trial[j];
                        preRepairTrialDelta[size_t(j)] =
                            (Trial[j] - PopulFront[TheChosenOne][j]) / material_span;
                        if (!use_tr) {
                            if (Trial[j] < Left) {
                                const double repaired = Random(Left, Right);
                                Trial[j] = repaired;
                                repairApplied[size_t(j)] = 1.0;
                                repairCandidateDelta[size_t(j)] =
                                    (repaired - PopulFront[TheChosenOne][j]) / material_span;
                            }
                            if (Trial[j] > Right) {
                                const double repaired = Random(Left, Right);
                                Trial[j] = repaired;
                                repairApplied[size_t(j)] = 1.0;
                                repairCandidateDelta[size_t(j)] =
                                    (repaired - PopulFront[TheChosenOne][j]) / material_span;
                            }
                        }
                        ActualCr++;
                    } else {
                        if (perturbation) {
                            double perturbNoise = material_nan;
                            const double perturbed = CachyRandWithNoise(PopulFront[TheChosenOne][j],
                                                                         F_cauchy_sigma_global_sop,
                                                                         perturbNoise);
                            Trial[j] = perturbed;
                            perturbationCauchyNoise[size_t(j)] = perturbNoise;
                            perturbationCandidateDelta[size_t(j)] =
                                (perturbed - PopulFront[TheChosenOne][j]) / material_span;
                        } else {
                            Trial[j] = PopulFront[TheChosenOne][j];
                        }
                        preRepairTrialX[size_t(j)] = Trial[j];
                        preRepairTrialDelta[size_t(j)] =
                            (Trial[j] - PopulFront[TheChosenOne][j]) / material_span;
                    }
                    if (!use_tr) {
                        if (Trial[j] < Left) {
                            const double repaired = Random(Left, Right);
                            Trial[j] = repaired;
                            repairApplied[size_t(j)] = 1.0;
                            repairCandidateDelta[size_t(j)] =
                                (repaired - PopulFront[TheChosenOne][j]) / material_span;
                        }
                        if (Trial[j] > Right) {
                            const double repaired = Random(Left, Right);
                            Trial[j] = repaired;
                            repairApplied[size_t(j)] = 1.0;
                            repairCandidateDelta[size_t(j)] =
                                (repaired - PopulFront[TheChosenOne][j]) / material_span;
                        }
                    }
	                }
	                xai_apply_init_guidance(Trial, NVars, Left, Right);
                    if (use_tr) {
                        RepairTrialInTrustRegion();
                    }
	            } else {
	                std::vector<char> crossoverMask;
                if (WeightedCrossoverEnabled_global_sop) {
                    int target = int(std::round(Cr * double(NVars)));
                    target = std::max(1, std::min(target, NVars));

                    std::vector<double> weights = DimensionWeights_global_sop;
                    if ((int)weights.size() != NVars) {
                        weights.assign(NVars, 1.0);
                    }
                    double sumw = 0.0;
                    for (int j = 0; j < NVars; ++j) {
                        double w = weights[j];
                        if (!std::isfinite(w) || w < 0.0) w = 0.0;
                        weights[j] = w;
                        sumw += w;
                    }
                    if (!(sumw > 0.0)) {
                        std::fill(weights.begin(), weights.end(), 1.0);
                    }

                    std::discrete_distribution<int> dimSelector(weights.begin(), weights.end());
                    crossoverMask.assign(NVars, 0);
                    crossoverMask[WillCrossover] = 1;
                    int filled = 1;
                    while (filled < target) {
                        int idx = dimSelector(generator_uni_i_3);
                        if (!crossoverMask[idx]) {
                            crossoverMask[idx] = 1;
                            filled++;
                        }
                    }
                }

                perturbation = LocalPerturbationEnabledWithUniform(perturbationUniform);
                std::vector<char> policyMask;
                const bool hasPolicyMask = maybe_apply_policy_control_callback(*this,
                                                                                NFEval,
                                                                                TheChosenOne,
                                                                                prand,
                                                                                Rand1,
                                                                                Rand2,
                                                                                use_eb,
                                                                                WillCrossover,
                                                                                meanF,
                                                                                sigmaF,
                                                                                psizeval,
                                                                                psizeval2,
                                                                                F,
                                                                                Cr,
                                                                                perturbation,
                                                                                policyMask);
                if (use_eb) {
                    EB_order(prand, Rand1, Rand2);
                }
                EB_hybrid_flag[TheChosenOne] = use_eb ? 1 : 0;
                for (int j = 0; j != NVars; j++) {
                    bool doCross = false;
                    if (hasPolicyMask) {
                        doCross = policyMask[size_t(j)] != 0;
                    } else if (WeightedCrossoverEnabled_global_sop) {
                        doCross = crossoverMask[j] != 0;
                    } else {
                        const double cross_u = Random(0, 1);
                        crossoverUniform[size_t(j)] = cross_u;
                        doCross = cross_u < Cr || WillCrossover == j;
                    }
                    realizedMask[size_t(j)] = doCross ? 1 : 0;
                    if (doCross) {
                        if (use_eb) {
                            Trial[j] = EbDonorValue(PopulFront[TheChosenOne][j],
                                                    ord_best_arch[j],
                                                    ord_medium_arch[j],
                                                    ord_worst_arch[j],
                                                    PopulFront[Rand1][j],
                                                    Popul[Rand2][j],
                                                    F);
                        } else {
                            Trial[j] = PopulFront[TheChosenOne][j] +
                                       F * (Popul[prand][j] - PopulFront[TheChosenOne][j]) +
                                       F * (PopulFront[Rand1][j] - Popul[Rand2][j]);
                        }
                        preRepairTrialX[size_t(j)] = Trial[j];
                        preRepairTrialDelta[size_t(j)] =
                            (Trial[j] - PopulFront[TheChosenOne][j]) / material_span;
                        if (!use_tr) {
                            if (Trial[j] < Left) {
                                const double repaired = Random(Left, Right);
                                Trial[j] = repaired;
                                repairApplied[size_t(j)] = 1.0;
                                repairCandidateDelta[size_t(j)] =
                                    (repaired - PopulFront[TheChosenOne][j]) / material_span;
                            }
                            if (Trial[j] > Right) {
                                const double repaired = Random(Left, Right);
                                Trial[j] = repaired;
                                repairApplied[size_t(j)] = 1.0;
                                repairCandidateDelta[size_t(j)] =
                                    (repaired - PopulFront[TheChosenOne][j]) / material_span;
                            }
                        }
                        ActualCr++;
                    } else {
                        if (perturbation) {
                            double perturbNoise = material_nan;
                            const double perturbed = CachyRandWithNoise(PopulFront[TheChosenOne][j],
                                                                         F_cauchy_sigma_global_sop,
                                                                         perturbNoise);
                            Trial[j] = perturbed;
                            perturbationCauchyNoise[size_t(j)] = perturbNoise;
                            perturbationCandidateDelta[size_t(j)] =
                                (perturbed - PopulFront[TheChosenOne][j]) / material_span;
                        } else {
                            Trial[j] = PopulFront[TheChosenOne][j];
                        }
                        preRepairTrialX[size_t(j)] = Trial[j];
                        preRepairTrialDelta[size_t(j)] =
                            (Trial[j] - PopulFront[TheChosenOne][j]) / material_span;
                    }
                    if (!use_tr) {
                        if (Trial[j] < Left) {
                            const double repaired = Random(Left, Right);
                            Trial[j] = repaired;
                            repairApplied[size_t(j)] = 1.0;
                            repairCandidateDelta[size_t(j)] =
                                (repaired - PopulFront[TheChosenOne][j]) / material_span;
                        }
                        if (Trial[j] > Right) {
                            const double repaired = Random(Left, Right);
                            Trial[j] = repaired;
                            repairApplied[size_t(j)] = 1.0;
                            repairCandidateDelta[size_t(j)] =
                                (repaired - PopulFront[TheChosenOne][j]) / material_span;
                        }
                    }
	                }
	                xai_apply_init_guidance(Trial, NVars, Left, Right);
                    if (use_tr) {
                        RepairTrialInTrustRegion();
                    }
	            }
	
	            if (maybe_apply_open_action_callback(*this,
	                                                 NFEval,
	                                                 TheChosenOne,
	                                                 prand,
	                                                 Rand1,
	                                                 Rand2,
	                                                 use_eb,
	                                                 perturbation,
	                                                 WillCrossover,
	                                                 MemoryCurrentIndex,
	                                                 MemoryCurrentIndex2,
	                                                 Rand_EB,
	                                                 meanF,
	                                                 sigmaF,
	                                                 psizeval,
	                                                 psizeval2,
	                                                 F,
	                                                 Cr,
	                                                 normalFNoise,
	                                                 normalFRejections,
	                                                 normalCrNoise,
	                                                 ebFNoise,
	                                                 ebFRejections,
	                                                 ebCrNoise,
	                                                 perturbationUniform,
	                                                 crossoverUniform,
	                                                 perturbationCandidateDelta,
	                                                 repairCandidateDelta,
	                                                 repairApplied,
	                                                 perturbationCauchyNoise,
	                                                 preRepairTrialDelta,
	                                                 preRepairTrialX,
	                                                 Trial,
	                                                 realizedMask,
	                                                 ActualCr)) {
	                if (!use_tr) {
	                    for (int j = 0; j < NVars; ++j) {
	                        if (Trial[j] < Left)
	                            Trial[j] = Random(Left, Right);
	                        if (Trial[j] > Right)
	                            Trial[j] = Random(Left, Right);
	                    }
	                }
	                xai_apply_init_guidance(Trial, NVars, Left, Right);
	                if (use_tr) {
	                    RepairTrialInTrustRegion();
	                }
	            }

	            ActualCr = ActualCr / double(NVars);
	            const int nfe_before_trial_eval = NFEval;
	            double TempFit = cec_24(Trial, func_num);
	            maybe_log_imitation_example(*this,
	                                        nfe_before_trial_eval,
	                                        TheChosenOne,
	                                        prand,
	                                        Rand1,
	                                        Rand2,
	                                        use_eb,
	                                        perturbation,
	                                        use_tr,
	                                        F,
	                                        Cr,
	                                        ActualCr,
	                                        WillCrossover,
	                                        MemoryCurrentIndex,
	                                        MemoryCurrentIndex2,
	                                        Rand_EB,
	                                        meanF,
	                                        sigmaF,
	                                        psizeval,
	                                        psizeval2,
	                                        normalFNoise,
	                                        normalFRejections,
	                                        normalCrNoise,
	                                        ebFNoise,
	                                        ebFRejections,
	                                        ebCrNoise,
	                                        perturbationUniform,
	                                        realizedMask,
	                                        crossoverUniform,
	                                        perturbationCandidateDelta,
	                                        repairCandidateDelta,
	                                        repairApplied,
	                                        perturbationCauchyNoise,
	                                        preRepairTrialDelta,
	                                        preRepairTrialX,
	                                        Trial,
	                                        TempFit);
	            if (XaiInternalEnabled_global_sop) {
	                xai_push_sample(Trial, TempFit, NVars);
	                xai_maybe_update_weights(NVars);
	            }
	            FitTemp[TheChosenOne] = TempFit;
	            if (TempFit <= FitArrFront[TheChosenOne]) {
                for (int j = 0; j != NVars; j++) {
                    Popul[NIndsCurrent + SuccessFilled][j] = Trial[j];
                    PopulFront[PFIndex][j] = Trial[j];
                }
                FitArr[NIndsCurrent + SuccessFilled] = TempFit;
                FitArrFront[PFIndex] = TempFit;
                FindNSaveBest(false, NIndsCurrent + SuccessFilled);
                tempSuccessCr[SuccessFilled] = ActualCr;
                tempSuccessF[SuccessFilled] = F;
                FitDelta[SuccessFilled] = fabs(FitArrFront[TheChosenOne] - TempFit);
                SuccessFilled++;
                PFIndex = (PFIndex + 1) % NIndsFront;
            }
            SaveBestValues(func_index);
        }
        for (int ChosenOne = 0; ChosenOne != NIndsFront; ChosenOne++) {
            FitTemp_prand[ChosenOne] = FitTemp[ChosenOne];
        }
        UpdateEB_hybrid_param(EB_hybrid_flag, FitMass, FitTemp_prand);
        for (int ChosenOne = 0; ChosenOne != NIndsFront; ChosenOne++) {
            FitMass[ChosenOne] = FitArrFront[ChosenOne];
        }
        SuccessRate = double(SuccessFilled) / double(NIndsFront);
        newNIndsFront =
            int(double(4 - NIndsFrontMax) / double(MaxFEval) * NFEval + NIndsFrontMax);
        RemoveWorst(NIndsFront, newNIndsFront);
        NIndsFront = newNIndsFront;
        UpdateMemoryCr();
        NIndsCurrent = NIndsFront + SuccessFilled;
        SuccessFilled = 0;
        UpdateTrustRegionAfterGeneration(generation_best_before);
        Generation++;
        if (GenerationHook_global_sop != nullptr && GenerationHook_global_sop(this)) {
            break;
        }
        if (NIndsCurrent > NIndsFront) {
            minfit = FitArr[0];
            maxfit = FitArr[0];
            for (int i = 0; i != NIndsCurrent; i++) {
                Indices[i] = i;
                maxfit = max(maxfit, FitArr[i]);
                minfit = min(minfit, FitArr[i]);
            }
            if (minfit != maxfit)
                qSort2int(FitArr, Indices, 0, NIndsCurrent - 1);
            NIndsCurrent = NIndsFront;
            for (int i = 0; i != NIndsCurrent; i++)
                for (int j = 0; j != NVars; j++)
                    PopulTemp[i][j] = Popul[Indices[i]][j];
            for (int i = 0; i != NIndsCurrent; i++)
                for (int j = 0; j != NVars; j++)
                    Popul[i][j] = PopulTemp[i][j];
        }
        
        if (HistoryCollectionEnabled_global_sop && g_snapshot_fe < 0 && NIndsCurrent > 10) {
            double v_mean = 0.0;
            int check_n = 10;
            for(int i=0; i<check_n; i++) v_mean += FitArr[i];
            v_mean /= check_n;
            double v_var = 0.0;
            for(int i=0; i<check_n; i++) {
                double d = FitArr[i] - v_mean;
                v_var += d*d;
            }
            if (v_var < 1e-12) {
                g_snapshot_fe = NFEval;
                for(int i=0; i<NIndsCurrent; i++) {
                    g_snapshot_f.push_back(FitArr[i]);
                    for(int j=0; j<NVars; j++) g_snapshot_x.push_back(Popul[i][j]);
                }
            }
        }
    }
}

void Optimizer::MainCycleSyncParallel(int parallel_workers, int small_batch_threshold) {
    if (!UseCec2017Evaluator_global_sop) {
        throw std::runtime_error("MainCycleSyncParallel requires pure CEC2017 evaluator mode");
    }

    vector<double> FitTemp2;
    vector<double> FitTemp_prand;
    std::vector<double> trial_matrix(size_t(NIndsFrontMax) * size_t(NVars));
    std::vector<double> trial_fit;
    std::vector<int> trial_chosen(size_t(NIndsFrontMax), 0);
    std::vector<int> trial_eb_flag(size_t(NIndsFrontMax), 0);
    std::vector<double> trial_actual_cr(size_t(NIndsFrontMax), 0.0);
    std::vector<double> trial_f(size_t(NIndsFrontMax), 0.0);
    trial_fit.reserve(size_t(NIndsFrontMax));

    evaluate_cec2017_rows_parallel(
        Popul,
        NIndsFront,
        NVars,
        func_num,
        FitArr,
        choose_generation_workers(parallel_workers, NIndsFront, small_batch_threshold));
    NFEval += NIndsFront;
    for (int IndIter = 0; IndIter < NIndsFront; IndIter++) {
        if (XaiInternalEnabled_global_sop) {
            xai_push_sample(Popul[IndIter], FitArr[IndIter], NVars);
            xai_maybe_update_weights(NVars);
        }
        FindNSaveBest(IndIter == 0, IndIter);
        if (!globalbestinit || bestfit < globalbest) {
            globalbest = bestfit;
            globalbestinit = true;
        }
    }
    SaveBestValues(func_index);

    double minfit = FitArr[0];
    double maxfit = FitArr[0];
    for (int i = 0; i != NIndsFront; i++) {
        FitArrCopy[i] = FitArr[i];
        Indices[i] = i;
        maxfit = max(maxfit, FitArr[i]);
        minfit = min(minfit, FitArr[i]);
    }
    if (minfit != maxfit)
        qSort2int(FitArrCopy, Indices, 0, NIndsFront - 1);
    for (int i = 0; i != NIndsFront; i++) {
        for (int j = 0; j != NVars; j++)
            PopulFront[i][j] = Popul[Indices[i]][j];
        FitArrFront[i] = FitArrCopy[i];
        FitMass[i] = FitArrFront[i];
    }

    PFIndex = 0;
    while (NFEval < MaxFEval) {
        const int evals_remaining = MaxFEval - NFEval;
        const int batch_count = std::min(NIndsFront, evals_remaining);
        if (batch_count <= 0) break;

        double meanF = 0.4 + tanh(SuccessRate * 5) * 0.25;
        double sigmaF = F_sigma_global_sop;

        minfit = FitArr[0];
        maxfit = FitArr[0];
        for (int i = 0; i != NIndsFront; i++) {
            FitArrCopy[i] = FitArr[i];
            Indices[i] = i;
            maxfit = max(maxfit, FitArr[i]);
            minfit = min(minfit, FitArr[i]);
        }
        if (minfit != maxfit)
            qSort2int(FitArrCopy, Indices, 0, NIndsFront - 1);

        minfit = FitArrFront[0];
        maxfit = FitArrFront[0];
        for (int i = 0; i != NIndsFront; i++) {
            FitArrCopy[i] = FitArrFront[i];
            Indices2[i] = i;
            maxfit = max(maxfit, FitArrFront[i]);
            minfit = min(minfit, FitArrFront[i]);
        }
        if (minfit != maxfit)
            qSort2int(FitArrCopy, Indices2, 0, NIndsFront - 1);

        FitTemp2.resize(NIndsFront);
        for (int i = 0; i != NIndsFront; i++)
            FitTemp2[i] = exp(-double(i) / double(NIndsFront) * 3.0);
        std::discrete_distribution<int> ComponentSelectorFront(FitTemp2.begin(), FitTemp2.end());

        int prand = 0;
        int Rand1 = 0;
        int Rand2 = 0;
        int psizeval = max(
            2,
            int(NIndsFront * PsizeCoef_global_sop * exp(-SuccessRate * SelectExp_global_sop))
        );
        FitTemp_prand.resize(NIndsFront);
        for (int i = 0; i != NIndsFront; i++)
            FitTemp_prand[i] = 3.0 * (NIndsFront - i);
        int psizeval2 =
            int(NIndsFront * 0.17 * (1 - 0.5 * (double)NFEval / (double)MaxFEval));
        if (psizeval2 <= 1)
            psizeval2 = 2;
        std::discrete_distribution<int> ComponentSelectorFront2(
            FitTemp_prand.begin(), FitTemp_prand.begin() + psizeval2);
        std::discrete_distribution<int> ComponentSelectorFront3(FitTemp_prand.begin(),
                                                                FitTemp_prand.end());

        for (int i = 0; i < NIndsFront; ++i) {
            FitMass[i] = FitArrFront[i];
            FitTemp[i] = FitArrFront[i];
            EB_hybrid_flag[i] = 0.0;
        }

        for (int IndIter = 0; IndIter < batch_count; IndIter++) {
            TheChosenOne = IntRandom(NIndsFront);
            MemoryCurrentIndex = IntRandom(MemorySize);
            MemoryCurrentIndex2 = IntRandom(MemorySize + 1);
            do
                prand = Indices[IntRandom(psizeval)];
            while (prand == TheChosenOne);
            do
                Rand1 = Indices2[ComponentSelectorFront(generator_uni_i_3)];
            while (Rand1 == prand);
            do
                Rand2 = Indices[IntRandom(NIndsFront)];
            while (Rand2 == prand || Rand2 == Rand1);
            do
                F = NormRand(meanF, sigmaF);
            while (F < 0.0 || F > 1.0);
            Cr = NormRand(MemoryCr[MemoryCurrentIndex], 0.05);
            Cr = min(max(Cr, 0.0), 1.0);
            double ActualCr = 0;
            int WillCrossover = IntRandom(NVars);

            double Rand_EB = Random(0, 1);
            if ((double)NFEval / (double)MaxFEval < EB_early_disable_frac_global_sop) {
                Rand_EB = EB_rand_multiplier_global_sop;
            }
            bool use_eb = (Rand_EB * (1 - double(NFEval) / double(MaxFEval))) < EB_hybrid_rate;
            if (use_eb) {
                do
                    prand = Indices[ComponentSelectorFront2(generator_uni_i_3)];
                while (prand == TheChosenOne);
                do
                    Rand1 = Indices2[ComponentSelectorFront3(generator_uni_i_3)];
                while (Rand1 == prand);
                do
                    Rand2 = Indices[ComponentSelectorFront3(generator_uni_i_3)];
                while (Rand2 == prand || Rand2 == Rand1);
                EB_order(prand, Rand1, Rand2);
                do {
                    if (MemoryCurrentIndex2 < MemorySize)
                        F = CachyRand(MemoryF[MemoryCurrentIndex2], F_cauchy_sigma_global_sop);
                    else
                        F = CachyRand(F_cauchy_fallback_mu_global_sop, F_cauchy_sigma_global_sop);
                } while (F < 0.0);
                if (F > 1.0)
                    F = 1.0;

                if ((double)NFEval / (double)MaxFEval < 0.6 && F > 0.7)
                    F = 0.7;

                if (MemoryCurrentIndex2 < MemorySize) {
                    if (MemoryCr[MemoryCurrentIndex2] < 0)
                        Cr = 0;
                    else
                        Cr = NormRand(MemoryCr[MemoryCurrentIndex2], Cr_sigma_global_sop);
                } else
                    Cr = NormRand(Cr_fallback_mu_global_sop, Cr_sigma_global_sop);
                if (Cr >= 1)
                    Cr = 1;
                if (Cr <= 0)
                    Cr = 0;

                if ((double)NFEval / (double)MaxFEval < 0.25)
                    Cr = max(Cr, Cr_early_min1_global_sop);
                if ((double)NFEval / (double)MaxFEval < 0.5)
                    Cr = max(Cr, Cr_early_min2_global_sop);
            }

            std::vector<char> crossoverMask;
            if (WeightedCrossoverEnabled_global_sop) {
                int target = int(std::round(Cr * double(NVars)));
                target = std::max(1, std::min(target, NVars));

                std::vector<double> weights = DimensionWeights_global_sop;
                if ((int)weights.size() != NVars) {
                    weights.assign(NVars, 1.0);
                }
                double sumw = 0.0;
                for (int j = 0; j < NVars; ++j) {
                    double w = weights[j];
                    if (!std::isfinite(w) || w < 0.0) w = 0.0;
                    weights[j] = w;
                    sumw += w;
                }
                if (!(sumw > 0.0)) {
                    std::fill(weights.begin(), weights.end(), 1.0);
                }

                std::discrete_distribution<int> dimSelector(weights.begin(), weights.end());
                crossoverMask.assign(NVars, 0);
                crossoverMask[WillCrossover] = 1;
                int filled = 1;
                while (filled < target) {
                    int idx = dimSelector(generator_uni_i_3);
                    if (!crossoverMask[idx]) {
                        crossoverMask[idx] = 1;
                        filled++;
                    }
                }
            }

            bool perturbation = LocalPerturbationEnabled();
            for (int j = 0; j != NVars; j++) {
                const bool doCross = WeightedCrossoverEnabled_global_sop
                    ? (crossoverMask[j] != 0)
                    : (Random(0, 1) < Cr || WillCrossover == j);
                if (doCross) {
                    if (use_eb) {
                        Trial[j] = EbDonorValue(PopulFront[TheChosenOne][j],
                                                ord_best_arch[j],
                                                ord_medium_arch[j],
                                                ord_worst_arch[j],
                                                PopulFront[Rand1][j],
                                                Popul[Rand2][j],
                                                F);
                    } else {
                        Trial[j] = PopulFront[TheChosenOne][j] +
                                   F * (Popul[prand][j] - PopulFront[TheChosenOne][j]) +
                                   F * (PopulFront[Rand1][j] - Popul[Rand2][j]);
                    }
                    if (Trial[j] < Left)
                        Trial[j] = Random(Left, Right);
                    if (Trial[j] > Right)
                        Trial[j] = Random(Left, Right);
                    ActualCr++;
                } else {
                    Trial[j] = perturbation
                                   ? CachyRand(PopulFront[TheChosenOne][j],
                                               F_cauchy_sigma_global_sop)
                                   : PopulFront[TheChosenOne][j];
                }
                if (Trial[j] < Left)
                    Trial[j] = Random(Left, Right);
                if (Trial[j] > Right)
                    Trial[j] = Random(Left, Right);
            }
            xai_apply_init_guidance(Trial, NVars, Left, Right);

            ActualCr = ActualCr / double(NVars);
            trial_chosen[size_t(IndIter)] = TheChosenOne;
            trial_eb_flag[size_t(IndIter)] = use_eb ? 1 : 0;
            trial_actual_cr[size_t(IndIter)] = ActualCr;
            trial_f[size_t(IndIter)] = F;
            for (int j = 0; j < NVars; ++j) {
                trial_matrix[size_t(IndIter) * size_t(NVars) + size_t(j)] = Trial[j];
            }
        }

        evaluate_cec2017_flat_parallel(
            trial_matrix,
            batch_count,
            NVars,
            func_num,
            trial_fit,
            choose_generation_workers(parallel_workers, batch_count, small_batch_threshold));
        NFEval += batch_count;

        for (int IndIter = 0; IndIter < batch_count; IndIter++) {
            const int chosen = trial_chosen[size_t(IndIter)];
            const double TempFit = trial_fit[size_t(IndIter)];
            FitTemp[chosen] = TempFit;
            EB_hybrid_flag[chosen] = double(trial_eb_flag[size_t(IndIter)]);

            if (XaiInternalEnabled_global_sop) {
                xai_push_sample(&trial_matrix[size_t(IndIter) * size_t(NVars)], TempFit, NVars);
                xai_maybe_update_weights(NVars);
            }

            if (TempFit <= FitMass[chosen]) {
                for (int j = 0; j != NVars; j++) {
                    const double value = trial_matrix[size_t(IndIter) * size_t(NVars) + size_t(j)];
                    Popul[NIndsCurrent + SuccessFilled][j] = value;
                    PopulFront[PFIndex][j] = value;
                }
                FitArr[NIndsCurrent + SuccessFilled] = TempFit;
                FitArrFront[PFIndex] = TempFit;
                FindNSaveBest(false, NIndsCurrent + SuccessFilled);
                tempSuccessCr[SuccessFilled] = trial_actual_cr[size_t(IndIter)];
                tempSuccessF[SuccessFilled] = trial_f[size_t(IndIter)];
                FitDelta[SuccessFilled] = fabs(FitMass[chosen] - TempFit);
                SuccessFilled++;
                PFIndex = (PFIndex + 1) % NIndsFront;
            }
        }

        if (!globalbestinit || bestfit < globalbest) {
            globalbest = bestfit;
            globalbestinit = true;
        }
        SaveBestValues(func_index);

        FitTemp_prand.resize(NIndsFront);
        for (int ChosenOne = 0; ChosenOne != NIndsFront; ChosenOne++) {
            FitTemp_prand[ChosenOne] = FitTemp[ChosenOne];
        }
        UpdateEB_hybrid_param(EB_hybrid_flag, FitMass, FitTemp_prand);
        for (int ChosenOne = 0; ChosenOne != NIndsFront; ChosenOne++) {
            FitMass[ChosenOne] = FitArrFront[ChosenOne];
        }
        SuccessRate = double(SuccessFilled) / double(NIndsFront);
        newNIndsFront =
            int(double(4 - NIndsFrontMax) / double(MaxFEval) * NFEval + NIndsFrontMax);
        RemoveWorst(NIndsFront, newNIndsFront);
        NIndsFront = newNIndsFront;
        UpdateMemoryCr();
        NIndsCurrent = NIndsFront + SuccessFilled;
        SuccessFilled = 0;
        Generation++;
        if (NIndsCurrent > NIndsFront) {
            minfit = FitArr[0];
            maxfit = FitArr[0];
            for (int i = 0; i != NIndsCurrent; i++) {
                Indices[i] = i;
                maxfit = max(maxfit, FitArr[i]);
                minfit = min(minfit, FitArr[i]);
            }
            if (minfit != maxfit)
                qSort2int(FitArr, Indices, 0, NIndsCurrent - 1);
            NIndsCurrent = NIndsFront;
            for (int i = 0; i != NIndsCurrent; i++)
                for (int j = 0; j != NVars; j++)
                    PopulTemp[i][j] = Popul[Indices[i]][j];
            for (int i = 0; i != NIndsCurrent; i++)
                for (int j = 0; j != NVars; j++)
                    Popul[i][j] = PopulTemp[i][j];
        }
        
        if (HistoryCollectionEnabled_global_sop && g_snapshot_fe < 0 && NIndsCurrent > 10) {
            double v_mean = 0.0;
            int check_n = 10;
            for(int i=0; i<check_n; i++) v_mean += FitArr[i];
            v_mean /= check_n;
            double v_var = 0.0;
            for(int i=0; i<check_n; i++) {
                double d = FitArr[i] - v_mean;
                v_var += d*d;
            }
            if (v_var < 1e-12) {
                g_snapshot_fe = NFEval;
                for(int i=0; i<NIndsCurrent; i++) {
                    g_snapshot_f.push_back(FitArr[i]);
                    for(int j=0; j<NVars; j++) g_snapshot_x.push_back(Popul[i][j]);
                }
            }
        }
    }
}

void Optimizer::Clean() {
    delete[] Trial;
    for (int i = 0; i != PopulSize; i++)
        delete[] Popul[i];
    for (int i = 0; i != NIndsFrontMax; i++)
        delete[] PopulFront[i];
    for (int i = 0; i != PopulSize; i++)
        delete[] PopulTemp[i];
    delete[] PopulTemp;
    delete[] Popul;
    delete[] PopulFront;
    delete[] FitArr;
    delete[] FitArrCopy;
    delete[] FitArrFront;
    delete[] Indices;
    delete[] Indices2;
    delete[] tempSuccessCr;
    delete[] tempSuccessF;
    delete[] FitDelta;
    delete[] MemoryCr;
    delete[] MemoryF;
    delete[] Weights;
    delete[] FitMass;
    delete[] EB_hybrid_flag;
    delete[] FitTemp;
    delete[] BestInd;
}

// ====== 平台接口实现 ======

OptimizationResultSOP run_rdex_sop_platform(
    int dimension,
    double lower_bound,
    double upper_bound,
    int population_size,
    int max_evaluations) {
    GNVars = dimension;
    MaxFEval = max_evaluations;
    Left_platform = lower_bound;
    Right_platform = upper_bound;

    if (XaiInternalEnabled_global_sop) {
        xai_reset_buffers(dimension);
    }

    globalbestinit = false;
    globalbest = std::numeric_limits<double>::infinity();
    LastFEcount = 0;
    NFEval = 0;

    for (int i = 0; i < ResTsize2; ++i) {
        ResultsArray[i] = 0.0;
    }

    Optimizer OptZ;
    OptZ.Initialize(population_size, dimension, /*func_num*/ 1, /*func_index*/ 1);
    OptZ.MainCycle();

    OptimizationResultSOP result;
    result.best_fitness = globalbest;
    // 与 RDEx-CSOP 保持一致：若内部循环略微超出 MaxFEval，这里在结果层做一次截断，
    // 确保平台侧看到的 evaluations_used 不超过配置预算。
    result.evaluations_used = (NFEval > MaxFEval) ? MaxFEval : NFEval;
    result.success = true;  // 无约束问题，始终视为成功
    result.best_solution.resize(dimension);
    for (int i = 0; i < dimension; ++i) {
        result.best_solution[i] = OptZ.BestInd ? OptZ.BestInd[i] : 0.0;
    }

    OptZ.Clean();

#ifdef RDEX_SOP_ENABLE_FINAL_PRINT
    // 可选的 C++ 端最终摘要输出（默认关闭，避免与平台重复）。
    // 如需调试对比精度，可在编译时定义 RDEX_SOP_ENABLE_FINAL_PRINT 宏启用。
    std::cout.setf(std::ios::scientific);
    std::cout.precision(18);
    std::cout << "[RDEx_SOP C++] final best_fitness=" << result.best_fitness
              << " evals=" << result.evaluations_used << std::endl;
#endif

    return result;
}

void set_cec2017_module_directory(const std::string& directory) {
    Cec2017ModuleDirectory_global_sop = directory;
    if (!directory.empty()) {
        cec17_set_input_data_dir((directory + "/input_data").c_str());
    }
}

std::vector<std::pair<int, double>> get_last_best_trace() {
    std::vector<std::pair<int, double>> out;
    out.reserve(size_t(ResTsize2 - 1));
    for (int k = 0; k < ResTsize2 - 1; ++k) {
        out.emplace_back(int(stepsFEval[k]), double(ResultsArray[k]));
    }
    return out;
}

std::vector<double> get_last_rbf_diagnostics() {
    return LastRBFDiagnostics_global_sop;
}

OptimizationResultSOP run_rdex_sop_cec2017_pure(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int population_size,
    int max_evaluations) {
    if (func_id < 1 || func_id > 30) {
        throw std::runtime_error("run_rdex_sop_cec2017_pure: func_id must be in [1, 30]");
    }
    cec17_reset_state();

    GNVars = dimension;
    MaxFEval = max_evaluations;
    Left_platform = lower_bound;
    Right_platform = upper_bound;

    if (XaiInternalEnabled_global_sop) {
        xai_reset_buffers(dimension);
    }

    globalbestinit = false;
    globalbest = std::numeric_limits<double>::infinity();
    LastFEcount = 0;
    NFEval = 0;

    for (int i = 0; i < ResTsize2; ++i) {
        ResultsArray[i] = 0.0;
    }

    Optimizer OptZ;
    Cec2017EvalModeGuard mode_guard(true);
    {
        WorkingDirectoryGuard wd_guard(Cec2017ModuleDirectory_global_sop);
        OptZ.Initialize(population_size, dimension, /*func_num*/ int(func_id), /*func_index*/ int(func_id));
        OptZ.MainCycle();
    }

    OptimizationResultSOP result;
    result.best_fitness = globalbest;
    // 与 RDEx-CSOP 保持一致：若内部循环略微超出 MaxFEval，这里在结果层做一次截断，
    // 确保平台侧看到的 evaluations_used 不超过配置预算。
    result.evaluations_used = (NFEval > MaxFEval) ? MaxFEval : NFEval;
    result.success = true;
    result.best_solution.resize(dimension);
    for (int i = 0; i < dimension; ++i) {
        result.best_solution[i] = OptZ.BestInd ? OptZ.BestInd[i] : 0.0;
    }

    OptZ.Clean();
    return result;
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    // Fixed Stage-12 candidate:
    // rde26-sop
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(false, 2.0, std::pow(0.5, 7), 2.0, 3, 5);
    set_trust_region_controller_params(0, 0.6, 1.0, 5);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trfull_gbest_st3_ft5_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    // Fixed S2 + full outer search trust region:
    // center=gbest, success_tolerance=3, failure_tolerance=5,
    // random repair only inside the current TR interval.
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 2.0, std::pow(0.5, 7), 2.0, 3, 5);
    set_trust_region_controller_params(0, 0.6, 1.0, 5);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate60_full_gbest_st3_ft5_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 2.0, std::pow(0.5, 7), 2.0, 3, 5);
    set_trust_region_controller_params(1, 0.6, 1.0, 5);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_full_gbest_st3_ft5_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 2.0, std::pow(0.5, 7), 2.0, 3, 5);
    set_trust_region_controller_params(1, 0.7, 1.0, 5);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_full_gbest_l1p0_st3_ft5_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 1.0, std::pow(0.5, 7), 2.0, 3, 5);
    set_trust_region_controller_params(1, 0.7, 1.0, 5);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_full_gbest_l1p0_st3_ft3_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 1.0, std::pow(0.5, 7), 2.0, 3, 3);
    set_trust_region_controller_params(1, 0.7, 1.0, 5);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_stag3_full_gbest_l1p0_st3_ft3_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 1.0, std::pow(0.5, 7), 2.0, 3, 3);
    set_trust_region_controller_params(3, 0.7, 1.0, 3);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_stag5_full_gbest_l1p0_st3_ft3_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 1.0, std::pow(0.5, 7), 2.0, 3, 3);
    set_trust_region_controller_params(3, 0.7, 1.0, 5);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_stag8_full_gbest_l1p0_st3_ft3_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 1.0, std::pow(0.5, 7), 2.0, 3, 3);
    set_trust_region_controller_params(3, 0.7, 1.0, 8);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_poststag3_full_gbest_l1p0_st3_ft3_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 1.0, std::pow(0.5, 7), 2.0, 3, 3);
    set_trust_region_controller_params(4, 0.7, 1.0, 3);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_poststag5_full_gbest_l1p0_st3_ft3_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 1.0, std::pow(0.5, 7), 2.0, 3, 3);
    set_trust_region_controller_params(4, 0.7, 1.0, 5);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_poststag8_full_gbest_l1p0_st3_ft3_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 1.0, std::pow(0.5, 7), 2.0, 3, 3);
    set_trust_region_controller_params(4, 0.7, 1.0, 8);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_mix50_gbest_l1p0_st3_ft3_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 1.0, std::pow(0.5, 7), 2.0, 3, 3);
    set_trust_region_controller_params(1, 0.7, 0.5, 5);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_mix25_gbest_l1p0_st3_ft3_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 1.0, std::pow(0.5, 7), 2.0, 3, 3);
    set_trust_region_controller_params(1, 0.7, 0.25, 5);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_detmix50_gbest_l1p0_st3_ft3_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 1.0, std::pow(0.5, 7), 2.0, 3, 3);
    set_trust_region_controller_params(1, 0.7, 0.5, 5);
    TrustRegionDeterministicMix_global_sop = true;

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_detmix25_gbest_l1p0_st3_ft3_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 1.0, std::pow(0.5, 7), 2.0, 3, 3);
    set_trust_region_controller_params(1, 0.7, 0.25, 5);
    TrustRegionDeterministicMix_global_sop = true;

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_full_gbest_l0p5_st3_ft3_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 0.5, std::pow(0.5, 7), 2.0, 3, 3);
    set_trust_region_controller_params(1, 0.7, 1.0, 5);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_full_gbest_l0p5_st3_ft5_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 0.5, std::pow(0.5, 7), 2.0, 3, 5);
    set_trust_region_controller_params(1, 0.7, 1.0, 5);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_mix50_gbest_l0p5_st3_ft5_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 0.5, std::pow(0.5, 7), 2.0, 3, 5);
    set_trust_region_controller_params(1, 0.7, 0.5, 5);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_mix25_gbest_l0p5_st3_ft5_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 0.5, std::pow(0.5, 7), 2.0, 3, 5);
    set_trust_region_controller_params(1, 0.7, 0.25, 5);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_detmix50_gbest_l0p5_st3_ft5_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 0.5, std::pow(0.5, 7), 2.0, 3, 5);
    set_trust_region_controller_params(1, 0.7, 0.5, 5);
    TrustRegionDeterministicMix_global_sop = true;

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_detmix25_gbest_l0p5_st3_ft5_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 0.5, std::pow(0.5, 7), 2.0, 3, 5);
    set_trust_region_controller_params(1, 0.7, 0.25, 5);
    TrustRegionDeterministicMix_global_sop = true;

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate70_full_gbest_l0p25_st3_ft7_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 0.25, std::pow(0.5, 7), 2.0, 3, 7);
    set_trust_region_controller_params(1, 0.7, 1.0, 5);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trlate65_full_gbest_l0p5_st3_ft5_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 0.5, std::pow(0.5, 7), 2.0, 3, 5);
    set_trust_region_controller_params(1, 0.65, 1.0, 5);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trstagnation5_full_gbest_st3_ft5_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 2.0, std::pow(0.5, 7), 2.0, 3, 5);
    set_trust_region_controller_params(2, 0.6, 1.0, 5);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_trmix25_always_gbest_st3_ft5_cec2017(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int max_evaluations) {
    set_memory_size(5);
    set_eb_hybrid_rate_init(0.7);
    set_original_eb_donor_enabled(true);
    set_cauchy_perturbation_prob(0.02);
    set_f_sigma(0.02);
    set_eb_f_cr_params(0.1, 0.9, 0.1, 0.9, 0.7, 0.6);
    set_eb_schedule_params(0.6, 2.0);
    set_select_pressure_params(7.0, 0.7);
    set_weighted_crossover_enabled(false);
    set_xai_internal_enabled(false);
    set_trust_region_params(true, 2.0, std::pow(0.5, 7), 2.0, 3, 5);
    set_trust_region_controller_params(0, 0.6, 0.25, 5);

    RbfInitialInjectionEnabled_global_sop = false;
    RbfRecordTrueSamples_global_sop = false;
    RbfSurrogateEvalMode_global_sop = false;
    GenerationHook_global_sop = nullptr;
    ActiveRBFModel_global_sop = nullptr;
    rbf_reset_archive();
    g_custom_init_pop.clear();

    return run_rdex_sop_cec2017_pure(
        dimension,
        func_id,
        lower_bound,
        upper_bound,
        435,
        max_evaluations);
}

OptimizationResultSOP run_rdex_sop_cec2017_rbf_initial(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int population_size,
    int max_evaluations,
    bool fast_subset) {
    if (func_id < 1 || func_id > 30) {
        throw std::runtime_error("run_rdex_sop_cec2017_rbf_initial: func_id must be in [1, 30]");
    }
    cec17_reset_state();

    GNVars = dimension;
    MaxFEval = max_evaluations;
    Left_platform = lower_bound;
    Right_platform = upper_bound;

    if (XaiInternalEnabled_global_sop) {
        xai_reset_buffers(dimension);
    }

    globalbestinit = false;
    globalbest = std::numeric_limits<double>::infinity();
    LastFEcount = 0;
    NFEval = 0;
    for (int i = 0; i < ResTsize2; ++i) {
        ResultsArray[i] = 0.0;
    }

    const bool saved_initial_enabled = RbfInitialInjectionEnabled_global_sop;
    const bool saved_record = RbfRecordTrueSamples_global_sop;
    const bool saved_surrogate = RbfSurrogateEvalMode_global_sop;
    const bool saved_fast_subset = RbfFastSubsetSelection_global_sop;
    RBFModelSOP* saved_active_model = ActiveRBFModel_global_sop;
    bool (*saved_hook)(Optimizer*) = GenerationHook_global_sop;

    RbfInitialInjectionEnabled_global_sop = true;
    RbfRecordTrueSamples_global_sop = true;
    RbfSurrogateEvalMode_global_sop = false;
    RbfFastSubsetSelection_global_sop = fast_subset;
    ActiveRBFModel_global_sop = nullptr;
    GenerationHook_global_sop = nullptr;
    rbf_reset_archive();

    Optimizer OptZ;
    Cec2017EvalModeGuard mode_guard(true);
    {
        WorkingDirectoryGuard wd_guard(Cec2017ModuleDirectory_global_sop);
        OptZ.Initialize(population_size, dimension, int(func_id), int(func_id));
        OptZ.MainCycle();
    }

    OptimizationResultSOP result;
    result.best_fitness = globalbest;
    result.evaluations_used = (NFEval > MaxFEval) ? MaxFEval : NFEval;
    result.success = true;
    result.best_solution.resize(dimension);
    for (int i = 0; i < dimension; ++i) {
        result.best_solution[i] = OptZ.BestInd ? OptZ.BestInd[i] : 0.0;
    }
    OptZ.Clean();
    RbfArchiveSamples_global_sop.clear();
    RbfArchiveFitness_global_sop.clear();

    RbfInitialInjectionEnabled_global_sop = saved_initial_enabled;
    RbfRecordTrueSamples_global_sop = saved_record;
    RbfSurrogateEvalMode_global_sop = saved_surrogate;
    RbfFastSubsetSelection_global_sop = saved_fast_subset;
    ActiveRBFModel_global_sop = saved_active_model;
    GenerationHook_global_sop = saved_hook;
    return result;
}

OptimizationResultSOP run_rdex_sop_cec2017_pure_parallel(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int population_size,
    int max_evaluations,
    int workers) {
    if (func_id < 1 || func_id > 30) {
        throw std::runtime_error("run_rdex_sop_cec2017_pure_parallel: func_id must be in [1, 30]");
    }
    cec17_reset_state();

    GNVars = dimension;
    MaxFEval = max_evaluations;
    Left_platform = lower_bound;
    Right_platform = upper_bound;

    if (XaiInternalEnabled_global_sop) {
        xai_reset_buffers(dimension);
    }

    globalbestinit = false;
    globalbest = std::numeric_limits<double>::infinity();
    LastFEcount = 0;
    NFEval = 0;

    for (int i = 0; i < ResTsize2; ++i) {
        ResultsArray[i] = 0.0;
    }

    Optimizer OptZ;
    Cec2017EvalModeGuard mode_guard(true);
    const int worker_count = resolve_parallel_workers(
        workers > 0 ? workers : ParallelWorkers_global_sop,
        population_size);
    {
        WorkingDirectoryGuard wd_guard(Cec2017ModuleDirectory_global_sop);
        OptZ.Initialize(population_size, dimension, /*func_num*/ int(func_id), /*func_index*/ int(func_id));
        OptZ.MainCycleSyncParallel(worker_count);
    }

    OptimizationResultSOP result;
    result.best_fitness = globalbest;
    result.evaluations_used = (NFEval > MaxFEval) ? MaxFEval : NFEval;
    result.success = true;
    result.best_solution.resize(dimension);
    for (int i = 0; i < dimension; ++i) {
        result.best_solution[i] = OptZ.BestInd ? OptZ.BestInd[i] : 0.0;
    }

    OptZ.Clean();
    return result;
}

OptimizationResultSOP run_rdex_sop_cec2017_pure_parallel_auto(
    int dimension,
    int func_id,
    double lower_bound,
    double upper_bound,
    int population_size,
    int max_evaluations,
    int max_workers,
    int small_batch_threshold) {
    if (func_id < 1 || func_id > 30) {
        throw std::runtime_error("run_rdex_sop_cec2017_pure_parallel_auto: func_id must be in [1, 30]");
    }
    cec17_reset_state();

    GNVars = dimension;
    MaxFEval = max_evaluations;
    Left_platform = lower_bound;
    Right_platform = upper_bound;

    if (XaiInternalEnabled_global_sop) {
        xai_reset_buffers(dimension);
    }

    globalbestinit = false;
    globalbest = std::numeric_limits<double>::infinity();
    LastFEcount = 0;
    NFEval = 0;

    for (int i = 0; i < ResTsize2; ++i) {
        ResultsArray[i] = 0.0;
    }

    Optimizer OptZ;
    Cec2017EvalModeGuard mode_guard(true);
    const int worker_count = resolve_parallel_workers(
        max_workers > 0 ? max_workers : ParallelWorkers_global_sop,
        population_size);
    int threshold = small_batch_threshold;
    if (threshold < 0) threshold = 0;
    {
        WorkingDirectoryGuard wd_guard(Cec2017ModuleDirectory_global_sop);
        OptZ.Initialize(population_size, dimension, /*func_num*/ int(func_id), /*func_index*/ int(func_id));
        OptZ.MainCycleSyncParallel(worker_count, threshold);
    }

    OptimizationResultSOP result;
    result.best_fitness = globalbest;
    result.evaluations_used = (NFEval > MaxFEval) ? MaxFEval : NFEval;
    result.success = true;
    result.best_solution.resize(dimension);
    for (int i = 0; i < dimension; ++i) {
        result.best_solution[i] = OptZ.BestInd ? OptZ.BestInd[i] : 0.0;
    }

    OptZ.Clean();
    return result;
}

// ====== 配置接口 ======

void set_random_seed(unsigned seed) {
    globalseed = seed;
    seed1 = globalseed;
    seed2 = globalseed + 100;
    seed3 = globalseed + 200;
    seed4 = globalseed + 300;
    seed5 = globalseed + 400;
    generator_uni_i.seed(seed1);
    generator_uni_r.seed(seed2);
    generator_norm.seed(seed3);
    generator_uni_i_3.seed(seed4);
    generator_cachy.seed(seed5);
    uni_int.reset();
    uni_real.reset();
    norm_dist.reset();
    cachy_dist.reset();
}

void set_parallel_workers(int workers) {
    ParallelWorkers_global_sop = workers;
}

void set_memory_size(int v) {
    if (v < 1) v = 1;
    if (v > 64) v = 64;
    // 当前实现中 MemorySize_default 固定为 5，真正的 MemorySize
    // 在 Optimizer::Initialize 中设置；平台版仅提供接口以便未来扩展。
    (void)v;
}

void set_eb_hybrid_rate_init(double v) {
    if (v < 0.0) v = 0.0;
    if (v > 1.0) v = 1.0;
    EB_hybrid_rate_init = v;
}

void set_original_eb_donor_enabled(bool enabled) {
    OriginalEbDonorEnabled_global_sop = enabled;
}

void set_original_perturbation_rand_enabled(bool enabled) {
    OriginalPerturbationRandEnabled_global_sop = enabled;
}

void set_c_rand_seed(unsigned seed) {
    std::srand(seed);
}

void set_cauchy_perturbation_prob(double v) {
    if (v < 0.0) v = 0.0;
    if (v > 1.0) v = 1.0;
    CauchyPerturbationProb_global_sop = v;
}

void set_f_sigma(double v) {
    if (v < 0.0) v = 0.0;
    F_sigma_global_sop = v;
}

void set_eb_f_cr_params(double f_cauchy_sigma,
                        double f_cauchy_fallback_mu,
                        double cr_sigma,
                        double cr_fallback_mu,
                        double cr_early_min1,
                        double cr_early_min2) {
    if (f_cauchy_sigma < 0.0) f_cauchy_sigma = 0.1;
    if (cr_sigma < 0.0) cr_sigma = 0.1;
    F_cauchy_sigma_global_sop = f_cauchy_sigma;
    F_cauchy_fallback_mu_global_sop = f_cauchy_fallback_mu;
    Cr_sigma_global_sop = cr_sigma;
    Cr_fallback_mu_global_sop = cr_fallback_mu;
    Cr_early_min1_global_sop = cr_early_min1;
    Cr_early_min2_global_sop = cr_early_min2;
}

void set_eb_schedule_params(double early_disable_frac, double rand_multiplier) {
    if (early_disable_frac < 0.0) early_disable_frac = 0.7;
    if (early_disable_frac > 1.0) early_disable_frac = 0.7;
    if (rand_multiplier <= 0.0) rand_multiplier = 2.0;
    EB_early_disable_frac_global_sop = early_disable_frac;
    EB_rand_multiplier_global_sop = rand_multiplier;
}

void set_select_pressure_params(double exp_k, double psize_coef) {
    if (exp_k <= 0.0) exp_k = 7.0;
    if (psize_coef <= 0.0) psize_coef = 0.7;
    SelectExp_global_sop = exp_k;
    PsizeCoef_global_sop = psize_coef;
}

void set_trust_region_params(bool enabled,
                             double length_init,
                             double length_min,
                             double length_max,
                             int success_tolerance,
                             int failure_tolerance) {
    TrustRegionEnabled_global_sop = enabled;
    if (!(length_init > 0.0)) length_init = 2.0;
    if (!(length_min > 0.0)) length_min = std::pow(0.5, 7);
    if (!(length_max > 0.0)) length_max = 2.0;
    if (length_min > length_max) std::swap(length_min, length_max);
    if (length_init < length_min) length_init = length_min;
    if (length_init > length_max) length_init = length_max;
    TrustRegionLengthInit_global_sop = length_init;
    TrustRegionLengthMin_global_sop = length_min;
    TrustRegionLengthMax_global_sop = length_max;
    TrustRegionSuccessTol_global_sop = std::max(1, success_tolerance);
    TrustRegionFailureTol_global_sop = std::max(1, failure_tolerance);
}

void set_trust_region_controller_params(int activation_mode,
                                        double late_start_frac,
                                        double mix_rate,
                                        int stagnation_generations) {
    if (activation_mode < 0 || activation_mode > 4) activation_mode = 0;
    if (!(late_start_frac >= 0.0 && late_start_frac <= 1.0)) late_start_frac = 0.6;
    if (!(mix_rate >= 0.0 && mix_rate <= 1.0)) mix_rate = 1.0;
    if (stagnation_generations < 1) stagnation_generations = 5;
    TrustRegionActivationMode_global_sop = activation_mode;
    TrustRegionLateStartFrac_global_sop = late_start_frac;
    TrustRegionMixRate_global_sop = mix_rate;
    TrustRegionStagnationTol_global_sop = stagnation_generations;
    TrustRegionDeterministicMix_global_sop = false;
}

void set_rbf_initial_params(int sample_cap,
                            int surrogate_patience,
                            int local_pool_cap) {
    if (sample_cap < 4) sample_cap = 4;
    if (sample_cap > 2000) sample_cap = 2000;
    if (surrogate_patience < 1) surrogate_patience = 1;
    if (surrogate_patience > 10000) surrogate_patience = 10000;
    if (local_pool_cap < sample_cap) local_pool_cap = sample_cap;
    if (local_pool_cap > 200000) local_pool_cap = 200000;
    RbfSampleCap_global_sop = sample_cap;
    RbfSurrogatePatience_global_sop = surrogate_patience;
    RbfLocalPoolCap_global_sop = local_pool_cap;
}

void set_rbf_fast_subset_enabled(bool enabled) {
    RbfFastSubsetSelection_global_sop = enabled;
}

void set_weighted_crossover_enabled(bool enabled) {
    WeightedCrossoverEnabled_global_sop = enabled;
}

void set_dimension_weights(const std::vector<double>& weights) {
    DimensionWeights_global_sop = weights;
}

void set_xai_internal_enabled(bool enabled) {
    XaiInternalEnabled_global_sop = enabled;
}

void set_xai_internal_params(int warmup_evals,
                             int update_interval,
                             int window_size,
                             double ridge_alpha,
                             int importance_mode,
                             bool use_rank_target,
                             double weight_floor,
                             double weight_smoothing,
                             double temperature) {
    if (warmup_evals < 0) warmup_evals = 0;
    if (update_interval < 1) update_interval = 1;
    if (window_size < 10) window_size = 10;
    XaiWarmupEvals_global_sop = warmup_evals;
    XaiUpdateInterval_global_sop = update_interval;
    XaiWindowSize_global_sop = window_size;
    XaiRidgeAlpha_global_sop = (ridge_alpha < 0.0) ? 0.0 : ridge_alpha;
    XaiImportanceMode_global_sop = importance_mode;
    XaiUseRankTarget_global_sop = use_rank_target;
    XaiWeightFloor_global_sop = (weight_floor < 0.0) ? 0.0 : weight_floor;
    XaiWeightSmoothing_global_sop = std::min(1.0, std::max(0.0, weight_smoothing));
    XaiTemperature_global_sop = (temperature > 0.0) ? temperature : 1.0;
}

void set_xai_init_params(int init_mode,
                         int init_phase_evals,
                         double init_prob,
                         int init_topk) {
    XaiInitMode_global_sop = init_mode;
    XaiInitPhaseEvals_global_sop = std::max(0, init_phase_evals);
    if (init_prob < 0.0) init_prob = 0.0;
    if (init_prob > 1.0) init_prob = 1.0;
    XaiInitProb_global_sop = init_prob;
    XaiInitTopK_global_sop = std::max(1, init_topk);
}

std::vector<double> get_dimension_weights() {
    return DimensionWeights_global_sop;
}

// History Collection Engine
void set_history_collection(bool enabled, double d_min) {
    HistoryCollectionEnabled_global_sop = enabled;
    HistoryArchiveDMin_global_sop = d_min;
}

void set_true_optimum_vector(const std::vector<double>& opt_x) {
    g_true_optimum_x = opt_x;
}

std::vector<double> get_history_diverse_archive_x() { return g_diverse_archive_x; }
std::vector<double> get_history_diverse_archive_f() { return g_diverse_archive_f; }
std::vector<double> get_history_trajectory_x() { return g_trajectory_x; }
std::vector<double> get_history_trajectory_f() { return g_trajectory_f; }
std::vector<int> get_history_trajectory_fe() { return g_trajectory_fe; }
std::vector<double> get_history_snapshot_x() { return g_snapshot_x; }
std::vector<double> get_history_snapshot_f() { return g_snapshot_f; }
int get_history_snapshot_fe() { return g_snapshot_fe; }
std::vector<double> get_history_nn_archive_x() { return g_nn_archive_x; }
std::vector<double> get_history_nn_archive_f() { return g_nn_archive_f; }
std::vector<double> get_history_nn_archive_dist() { return g_nn_archive_dist; }

void clear_history_collection() {
    g_diverse_archive_x.clear();
    g_diverse_archive_f.clear();
    g_trajectory_x.clear();
    g_trajectory_f.clear();
    g_trajectory_fe.clear();
    g_snapshot_x.clear();
    g_snapshot_f.clear();
    g_snapshot_fe = -1;
    g_nn_archive_x.clear();
    g_nn_archive_f.clear();
    g_nn_archive_dist.clear();
}

void set_imitation_logging(bool enabled, int max_examples, int stride) {
    ImitationLoggingEnabled_global_sop = enabled;
    ImitationMaxExamples_global_sop = std::max(0, max_examples);
    ImitationStride_global_sop = std::max(1, stride);
}

void clear_imitation_log() {
    ImitationSeenUpdates_global_sop = 0;
    g_imitation_status.clear();
    g_imitation_roles.clear();
    g_imitation_role_fitness.clear();
    g_imitation_mask.clear();
    g_imitation_delta.clear();
    g_imitation_trial_x.clear();
    g_imitation_action_meta.clear();
    g_imitation_int_meta.clear();
    g_imitation_memory_f.clear();
    g_imitation_memory_cr.clear();
    g_imitation_mechanism_meta.clear();
    g_imitation_vector_materials.clear();
}

std::vector<int> get_imitation_shape() {
    const int dim = GNVars;
    const int status_dim = int(ImitationStatusNames_global_sop.size());
    const int role_count = int(ImitationRoleNames_global_sop.size());
    const int action_meta_dim = int(ImitationActionMetaNames_global_sop.size());
    const int int_meta_dim = int(ImitationIntMetaNames_global_sop.size());
    int n = 0;
    if (dim > 0) {
        n = int(g_imitation_delta.size() / size_t(dim));
    }
    return {n, status_dim, role_count, dim, action_meta_dim, int_meta_dim};
}

std::vector<std::string> get_imitation_status_names() {
    return ImitationStatusNames_global_sop;
}

std::vector<std::string> get_imitation_role_names() {
    return ImitationRoleNames_global_sop;
}

std::vector<std::string> get_imitation_action_meta_names() {
    return ImitationActionMetaNames_global_sop;
}

std::vector<std::string> get_imitation_int_meta_names() {
    return ImitationIntMetaNames_global_sop;
}

std::vector<std::string> get_imitation_mechanism_meta_names() {
    return ImitationMechanismMetaNames_global_sop;
}

std::vector<std::string> get_imitation_vector_material_names() {
    return ImitationVectorMaterialNames_global_sop;
}

std::vector<double> get_imitation_status() {
    return g_imitation_status;
}

std::vector<double> get_imitation_roles() {
    return g_imitation_roles;
}

std::vector<double> get_imitation_role_fitness() {
    return g_imitation_role_fitness;
}

std::vector<double> get_imitation_mask() {
    return g_imitation_mask;
}

std::vector<double> get_imitation_delta() {
    return g_imitation_delta;
}

std::vector<double> get_imitation_trial_x() {
    return g_imitation_trial_x;
}

std::vector<double> get_imitation_action_meta() {
    return g_imitation_action_meta;
}

std::vector<int> get_imitation_int_meta() {
    return g_imitation_int_meta;
}

std::vector<double> get_imitation_memory_f() {
    return g_imitation_memory_f;
}

std::vector<double> get_imitation_memory_cr() {
    return g_imitation_memory_cr;
}

std::vector<double> get_imitation_mechanism_meta() {
    return g_imitation_mechanism_meta;
}

std::vector<double> get_imitation_vector_materials() {
    return g_imitation_vector_materials;
}

void set_policy_control_callback(PolicyControlCallbackSOP callback) {
    PolicyControlCallback_global_sop = std::move(callback);
    PolicyControlCallbackEnabled_global_sop = true;
}

void clear_policy_control_callback() {
    PolicyControlCallback_global_sop = nullptr;
    PolicyControlCallbackEnabled_global_sop = false;
}

bool is_policy_control_callback_enabled() {
    return PolicyControlCallbackEnabled_global_sop;
}

void set_open_action_callback(OpenActionCallbackSOP callback) {
    OpenActionCallback_global_sop = std::move(callback);
    OpenActionCallbackEnabled_global_sop = true;
}

void clear_open_action_callback() {
    OpenActionCallback_global_sop = nullptr;
    OpenActionCallbackEnabled_global_sop = false;
}

bool is_open_action_callback_enabled() {
    return OpenActionCallbackEnabled_global_sop;
}
