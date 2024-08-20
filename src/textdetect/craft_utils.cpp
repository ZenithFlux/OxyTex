#include "craft_utils.hpp"
#include <opencv2/opencv.hpp>
#include <climits>
#include <cmath>
#include <cstdint>
#include <algorithm>


namespace txdt {

struct DetBoxesData {
    cv::Mat_<cv::Point> det;
    cv::Mat_<int32_t> labels;
    std::vector<int> mapper;
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

    cv::Mat centroids;
    cv::Mat_<int32_t> stats, labels;
    int n_labels = cv::connectedComponentsWithStats(
        text_score_comb,
        labels,
        stats,
        centroids,
        4
    );

    int box_count = 0;
    cv::Mat_<cv::Point> det(n_labels, 4);
    std::vector<int> mapper;
    for (int k=1; k < n_labels; ++k) {
        // size filtering
        int size = stats(k, cv::CC_STAT_AREA);
        if (size < 10) continue;

        // thresholding
        double max_val;
        cv::minMaxIdx(textmap, nullptr, &max_val, nullptr, nullptr, labels == k);
        if (max_val < cfg.text_thr) continue;

        // make segmentation map
        cv::Mat segmap{
            textmap.rows,
            textmap.cols,
            CV_MAKETYPE(CV_8U, textmap.channels()),
            0
        };
        segmap.setTo(255, labels == k);
        cv::Mat link_area;
        cv::bitwise_and(link_score == 1, text_score == 0, link_area);
        segmap.setTo(0, link_area);
        int32_t &x = stats(k, cv::CC_STAT_LEFT), &y = stats(k, cv::CC_STAT_TOP);
        int32_t &w = stats(k, cv::CC_STAT_WIDTH), &h = stats(k, cv::CC_STAT_HEIGHT);
        int32_t niter = std::sqrt(size * std::min(w, h) / (w * h)) * 2;
        int32_t sx = std::max(x - niter, 0);
        int32_t ex = std::min(x + w + niter + 1, img_w);
        int32_t sy = std::max(y - niter, 0);
        int32_t ey = std::min(y + h + niter + 1, img_h);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {1+niter, 1+niter});
        cv::Mat submat = segmap(cv::Range(sy, ey), cv::Range(sx, ex));
        cv::dilate(submat, submat, kernel);

        // make box
        std::vector<cv::Point> contours, box;
        cv::findNonZero(segmap, contours);
        cv::RotatedRect rectangle = cv::minAreaRect(contours);
        cv::boxPoints(rectangle, box);

        // align diamond shape
        double wb = cv::norm(box[0] - box[1]);
        double hb = cv::norm(box[1] - box[2]);
        double box_ratio = std::max(wb, hb) / (std::min(wb, hb) + 1e-5);
        if (std::abs(1 - box_ratio) <= 0.1) {
            std::vector<int> ct_x(contours.size()), ct_y(contours.size());
            for (int i=0; i < contours.size(); ++i) {
                ct_x[i] = contours[i].x;
                ct_y[i] = contours[i].y;
            }
            int l = *std::min_element(ct_x.begin(), ct_x.end());
            int r = *std::max_element(ct_x.begin(), ct_x.end());
            int t = *std::min_element(ct_y.begin(), ct_y.end());
            int b = *std::max_element(ct_y.begin(), ct_y.end());
            box = {
                cv::Point(l, t),
                cv::Point(r, t),
                cv::Point(r, b),
                cv::Point(l, b)
            };
        }
        // make clockwise order
        size_t startidx = std::min_element(
            box.begin(),
            box.end(),
            [](cv::Point a, cv::Point b) { return a.x + a.y < b.x + b.y; }
        ) - box.begin();

        std::rotate(box.begin(), box.begin()+startidx, box.end());

        ++box_count;
        det.push_back(box);
        mapper.push_back(k);
    }
    det.resize(box_count);
    return {det, labels, mapper};
}

}

}
