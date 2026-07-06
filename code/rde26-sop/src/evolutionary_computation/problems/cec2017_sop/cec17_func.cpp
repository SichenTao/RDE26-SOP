#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

double *OShift = nullptr;
double *M = nullptr;
double *y = nullptr;
double *z = nullptr;
double *x_bound = nullptr;
int ini_flag = 0;
int n_flag = 0;
int func_flag = 0;
int *SS = nullptr;

namespace {

std::string g_input_data_dir;
std::mutex g_cec17_mutex;

bool starts_with(const char* text, const char* prefix) {
    return std::strncmp(text, prefix, std::strlen(prefix)) == 0;
}

FILE* cec17_bundle_fopen(const char* filename, const char* mode) {
    if (filename != nullptr && mode != nullptr && !g_input_data_dir.empty() &&
        starts_with(filename, "input_data/")) {
        const char* relative = filename + std::strlen("input_data/");
        std::string full_path = g_input_data_dir + "/" + relative;
        return std::fopen(full_path.c_str(), mode);
    }
    return std::fopen(filename, mode);
}

void free_cec17_state() {
    std::free(M);
    std::free(OShift);
    std::free(y);
    std::free(z);
    std::free(x_bound);
    std::free(SS);
    M = nullptr;
    OShift = nullptr;
    y = nullptr;
    z = nullptr;
    x_bound = nullptr;
    SS = nullptr;
}

}  // namespace

#define fopen cec17_bundle_fopen
#define cec17_test_func cec17_test_func_original
#include "cec17_test_func.cpp"
#undef cec17_test_func
#undef fopen

void cec17_set_input_data_dir(const char* dir) {
    std::lock_guard<std::mutex> lock(g_cec17_mutex);
    g_input_data_dir = (dir == nullptr) ? std::string() : std::string(dir);
    while (!g_input_data_dir.empty() &&
           (g_input_data_dir.back() == '/' || g_input_data_dir.back() == '\\')) {
        g_input_data_dir.pop_back();
    }
    ini_flag = 0;
}

void cec17_reset_state() {
    std::lock_guard<std::mutex> lock(g_cec17_mutex);
    free_cec17_state();
    ini_flag = 0;
    n_flag = 0;
    func_flag = 0;
}

void cec17_test_func(double* x, double* f, int nx, int mx, int func_num) {
    std::lock_guard<std::mutex> lock(g_cec17_mutex);
    cec17_test_func_original(x, f, nx, mx, func_num);
}
