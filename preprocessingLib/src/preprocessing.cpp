#include "preprocessingLib.h"

namespace preprocessing {

    cv::Mat imread(const std::string& filename, int flags) {
        return cv::imread(filename, flags);
    }

    cv::Mat resize(const cv::Mat& img, const cv::Size& size) {
        cv::Mat resized;
        cv::resize(img, resized, size);
        return resized;
    }

}
