#ifndef OXYTEX_TEXTDETECT_H_
#define OXYTEX_TEXTDETECT_H_

#include <string>
#include <opencv2/opencv.hpp>

namespace txdt {

cv::Mat load_image(const std::string& img_file);

std::pair<cv::Mat, double> resize_aspect_ratio(
    cv::Mat img,
    std::size_t square_size,
    int interpolation,
    double mag_ratio = 1
);

cv::Mat normalize_mean_variance(
    cv::Mat img,
    cv::Scalar mean = {0.485, 0.456, 0.406},
    cv::Scalar variance = {0.229, 0.224, 0.225}
);

}

#endif
