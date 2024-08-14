#include "detect.hpp"
#include "imgproc.hpp"
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>


namespace txdt {

using namespace torch::indexing;

// ----------------------------- CraftDetector ---------------------------------

CraftDetector::CraftDetector() {
    // Download models
}


CraftDetector::CraftDetector(
    const std::string& craft_path,
    const optional_ref<std::string> refine_path
) {
    _craftnet = torch::jit::load(craft_path);
    _craftnet.eval();
    if (refine_path) {
        _refinenet = torch::jit::load(refine_path.value());
        _refinenet.value().eval();
    }
}


CraftDetector::Output CraftDetector::detect_text(
    cv::Mat img,
    const DetectConfig& cfg,
    bool cuda
) {
    double target_ratio;
    std::tie(img, target_ratio) = resize_aspect_ratio(
        img,
        cfg.max_size,
        cv::INTER_LINEAR,
        cfg.mag_ratio
    );
    img = txdt::normalize_mean_variance(img);
    torch::Tensor img_ten = cv_to_torch(img, torch::dtype(torch::kF32));
    img_ten.unsqueeze(0);
    if (cuda) img_ten.to(torch::kCUDA);

    torch::NoGradGuard no_grad;
    c10::ivalue::TupleElements out = _craftnet.forward({img_ten}).toTuple()->elements();
    torch::Tensor &y = out[0].toTensor(), &feature = out[1].toTensor();

    torch::Tensor score_text = y.index({0, "...", 0}).detach().cpu();
    torch::Tensor score_link = y.index({0, "...", 1}).detach().cpu();

    if (_refinenet) {
        torch::Tensor y_refiner = _refinenet.value().forward({y, feature}).toTensor();
        score_link = y_refiner.index({0, "...", 0}).detach().cpu();
    }

    // Implement getDetBoxes
}

}
