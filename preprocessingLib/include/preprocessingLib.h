#ifndef PREPROCESSING_LIB_H
#define PREPROCESSING_LIB_H

#include <opencv2/opencv.hpp>
#include <string>

namespace preprocessingLib {
    cv::Mat imread(const std::string& filename, int flags = cv::IMREAD_COLOR);
}

#endif // PREPROCESSING_LIB_H