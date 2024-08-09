#include <algorithm>
#include <cstddef>
#include <string>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <utility>


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


cv::Mat normalize_mean_variance(cv::Mat img, cv::Scalar mean, cv::Scalar variance) {
    img.convertTo(img, CV_32F);
    img -= mean * 255;
    img /= variance * 255;
    return img;
};

}
