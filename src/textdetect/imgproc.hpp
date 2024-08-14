#ifndef OXYTEX_TEXTDETECT_IMGPROC_HPP_
#define OXYTEX_TEXTDETECT_IMGPROC_HPP_

#include <opencv2/opencv.hpp>
#include <torch/script.h>

namespace txdt {

cv::Mat load_image(const std::string& img_file);


std::pair<cv::Mat, double> resize_aspect_ratio(
    cv::Mat img,
    std::size_t max_size,
    int interpolation,
    double mag_ratio = 1
);


cv::Mat normalize_mean_variance(
    cv::Mat img,
    cv::Scalar mean = {0.485, 0.456, 0.406},
    cv::Scalar variance = {0.229, 0.224, 0.225}
);


torch::Tensor cv_to_torch(const cv::Mat& mat, const torch::TensorOptions opts);

/* @param type Mat type E.g. CV_8UC3.
   @param ten Tensor to be converted to Mat.
   First dimension of `ten` should be no. of channels. */
cv::Mat torch_to_cv(const torch::Tensor& ten, int type);

}

#endif
