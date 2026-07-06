#include "rdex_sop.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "usage: rde26_sop function_id seed max_evaluations dimension lower_bound upper_bound cec_dir\n";
        return 2;
    }

    const int function_id = std::atoi(argv[1]);
    const unsigned seed = static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10));
    const int max_evaluations = std::atoi(argv[3]);
    const int dimension = std::atoi(argv[4]);
    const double lower_bound = std::atof(argv[5]);
    const double upper_bound = std::atof(argv[6]);
    const std::string cec_dir = argv[7];

    try {
        set_cec2017_module_directory(cec_dir);
        set_random_seed(seed);
        run_rdex_sop_s2_xover_w0_ed60_rm2p0_c02_d1_cec2017(
            dimension,
            function_id,
            lower_bound,
            upper_bound,
            max_evaluations
        );
        for (const auto& item : get_last_best_trace()) {
            std::cout << item.first << '\t' << item.second << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
