#include <iostream>
#include <torch/script.h>


namespace textdetect {
    void load_models() {
        torch::jit::script::Module craft_net = torch::jit::load("models/textdetect/craft_mlt_25k.pt");
        std::cout << "Craftnet loaded!" << std::endl;
        torch::jit::script::Module refine_net = torch::jit::load("models/textdetect/craft_refiner_CTW1500.pt");
        std::cout << "Refinenet loaded!" << std::endl;
    }
}
