#include "imgproc.hpp"
#include <cstring>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <algorithm>
#include <vector>
#include <cstddef>
#include <cstdint>


namespace txdt {


cv::Mat load_image(const std::string& img_file) {
    cv::Mat img = cv::imread(img_file);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    return img;
}


std::pair<cv::Mat, double> resize_aspect_ratio(
    cv::Mat img,
    std::size_t square_size,
    int interpolation,
    double mag_ratio
) {
    int height = img.size[0], width = img.size[1];
    double target_size = mag_ratio * std::max(height, width);

    if (target_size > square_size) target_size = square_size;
    mag_ratio = target_size / std::max(height, width);

    int target_h = height * mag_ratio, target_w = width * mag_ratio;
    cv::resize(img, img, {target_w, target_h}, 0, 0, interpolation);

    int pad_b = 0, pad_r = 0;
    if (target_h % 32 != 0) {
        pad_b = 32 - target_h % 32;
    }
    if (target_w % 32 != 0) {
        pad_r = 32 - target_w % 32;
    }
    cv::copyMakeBorder(img, img, 0, pad_b, 0, pad_r, cv::BORDER_CONSTANT, 0);
    target_h += pad_b;
    target_w += pad_r;

    return {img, mag_ratio};
}


cv::Mat normalize_mean_variance(
    cv::Mat img,
    const cv::Scalar& mean,
    const cv::Scalar& variance
) {
    img.convertTo(img, CV_32F);
    img -= mean * 255;
    img /= variance * 255;
    return img;
};


torch::Tensor cv_to_torch(const cv::Mat& mat, const torch::TensorOptions opts) {
    std::vector<int64_t> sizes{mat.size.p, mat.size.p + mat.size.dims()};
    sizes.push_back(mat.channels());
    torch::Tensor ten = torch::empty(sizes, opts);
    std::memcpy(ten.mutable_data_ptr(), mat.data, mat.elemSize() * mat.total());
    return ten;
}


cv::Mat torch_to_cv(const torch::Tensor& ten, int type) {
    cv::Mat img{{ten.sizes().begin() + 1, ten.sizes().end()}, type};
    std::memcpy(img.data, ten.const_data_ptr(), ten.numel());
    return img;
}

}
