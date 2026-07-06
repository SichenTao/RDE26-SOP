#include "problem_bridge.h"

#include <mutex>
#include <stdexcept>

namespace problem_bridge {
namespace {

EvaluateCallback g_evaluate_callback;
ConstraintCallback g_constraint_callback;
DetailedConstraintCallback g_detailed_constraint_callback;
BoundsCallback g_bounds_callback;
std::mutex g_callback_mutex;

std::vector<double> make_vector(const double* x, int nvars) {
    if (x == nullptr || nvars < 0) {
        throw std::runtime_error("problem_bridge: invalid solution buffer");
    }
    return std::vector<double>(x, x + nvars);
}

}  // namespace

void set_evaluate_callback(EvaluateCallback callback) {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    g_evaluate_callback = std::move(callback);
}

void set_constraint_callback(ConstraintCallback callback) {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    g_constraint_callback = std::move(callback);
}

void set_detailed_constraint_callback(DetailedConstraintCallback callback) {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    g_detailed_constraint_callback = std::move(callback);
}

void set_bounds_callback(BoundsCallback callback) {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    g_bounds_callback = std::move(callback);
}

double evaluate_solution(const double* x, int nvars) {
    EvaluateCallback callback;
    {
        std::lock_guard<std::mutex> lock(g_callback_mutex);
        callback = g_evaluate_callback;
    }
    if (!callback) {
        throw std::runtime_error("problem_bridge: evaluation callback is not set");
    }
    return callback(make_vector(x, nvars), nvars);
}

double evaluate_solution(double* x, int nvars) {
    return evaluate_solution(static_cast<const double*>(x), nvars);
}

double constraint_violation(const double* x, int nvars) {
    ConstraintCallback callback;
    {
        std::lock_guard<std::mutex> lock(g_callback_mutex);
        callback = g_constraint_callback;
    }
    if (!callback) {
        return 0.0;
    }
    return callback(make_vector(x, nvars), nvars);
}

std::pair<std::vector<double>, std::vector<double>>
detailed_constraints(const double* x, int nvars) {
    DetailedConstraintCallback callback;
    {
        std::lock_guard<std::mutex> lock(g_callback_mutex);
        callback = g_detailed_constraint_callback;
    }
    if (!callback) {
        return {};
    }
    return callback(make_vector(x, nvars), nvars);
}

std::pair<std::vector<double>, std::vector<double>> get_bounds() {
    BoundsCallback callback;
    {
        std::lock_guard<std::mutex> lock(g_callback_mutex);
        callback = g_bounds_callback;
    }
    if (!callback) {
        return {};
    }
    return callback();
}

}  // namespace problem_bridge
