#include <opencv2/opencv.hpp>
#include <textdetect/detect.hpp>
#include <tuple>


int main() {
    cv::Mat img = txdt::load_image("build/test_files/test.jpg");
    double target_ratio;
    std::tie(img, target_ratio) = txdt::resize_aspect_ratio(img, 1280, cv::INTER_LINEAR, 1.5);
    img = txdt::normalize_mean_variance(img);
    return 0;
}
