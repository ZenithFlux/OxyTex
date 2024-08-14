#ifndef OXYTEX_TEXTDETECT_DETECT_HPP_
#define OXYTEX_TEXTDETECT_DETECT_HPP_

#include <string>
#include <opencv2/opencv.hpp>
#include <optional>
#include <torch/torch.h>

namespace txdt {

template <typename T>
using optional_ref = ::std::optional<std::reference_wrapper<T>>;

namespace TS = ::torch::jit::script;


struct DetectConfig {
    double mag_ratio;
    int max_size;
    double text_thr;
    double link_thr;
    double low_bound_text;
    bool poly;
};


class CraftDetector {

    public:
        struct Output {
            cv::Mat heatmap;
        };

        // Download models
        CraftDetector();

        // Load local models
        CraftDetector(
            const std::string& craft_path,
            const optional_ref<std::string> refine_path = std::nullopt
        );

        // Detect text in the given image
        Output detect_text(cv::Mat img, const DetectConfig& cfg, bool cuda);

    private:
        TS::Module _craftnet;
        std::optional<TS::Module> _refinenet;
};

}

#endif
