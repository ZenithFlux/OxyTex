#include "craft_utils.hpp"
#include <cstdint>
#include <opencv2/opencv.hpp>


namespace txdt {

struct DetBoxesData {

};

DetBoxesData get_det_boxes_core(
    const cv::Mat& textmap,
    const cv::Mat& linkmap,
    const DetectConfig& cfg
) {
    int img_h = textmap.rows, img_w = textmap.cols;
    cv::Mat text_score, link_score, text_score_comb;

    cv::threshold(textmap, text_score, cfg.low_bound_text, 1, 0);
    cv::threshold(linkmap, link_score, cfg.link_thr, 1, 0);
    text_score_comb = text_score + link_score;
    text_score_comb.setTo(0, text_score_comb < 0);
    text_score_comb.setTo(1, text_score_comb > 1);
    text_score_comb.convertTo(text_score_comb, CV_8U);

    cv::Mat labels, centroids;
    cv::Mat_<int32_t> stats;
    int n_labels = cv::connectedComponentsWithStats(
        text_score_comb,
        labels, stats,
        centroids,
        4
    );

    for (int k=1; k < n_labels; ++k) {
        int size = stats(k, cv::CC_STAT_AREA);
        if (size < 10) continue;

        double max_val;
        cv::minMaxIdx(textmap, nullptr, &max_val, nullptr, nullptr, labels == k);
        if (max_val < cfg.text_thr) continue;
        // Make segmentation map
    }
}

}
