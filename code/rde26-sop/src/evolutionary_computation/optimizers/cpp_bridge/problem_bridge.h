#pragma once

#include <functional>
#include <utility>
#include <vector>

namespace problem_bridge {

using EvaluateCallback = std::function<double(const std::vector<double>&, int)>;
using ConstraintCallback = std::function<double(const std::vector<double>&, int)>;
using DetailedConstraintCallback =
    std::function<std::pair<std::vector<double>, std::vector<double>>(
        const std::vector<double>&,
        int)>;
using BoundsCallback =
    std::function<std::pair<std::vector<double>, std::vector<double>>()>;

void set_evaluate_callback(EvaluateCallback callback);
void set_constraint_callback(ConstraintCallback callback);
void set_detailed_constraint_callback(DetailedConstraintCallback callback);
void set_bounds_callback(BoundsCallback callback);

double evaluate_solution(const double* x, int nvars);
double evaluate_solution(double* x, int nvars);
double constraint_violation(const double* x, int nvars);
std::pair<std::vector<double>, std::vector<double>>
detailed_constraints(const double* x, int nvars);
std::pair<std::vector<double>, std::vector<double>> get_bounds();

}  // namespace problem_bridge
