#ifndef PREPROCESSING_LIB_H
#define PREPROCESSING_LIB_H

#include <opencv2/opencv.hpp>

namespace preprocessing {

    cv::Mat imread(const std::string& filename, int flags = cv::IMREAD_COLOR);

    cv::Mat resize(const cv::Mat& img, const cv::Size& size);

} 

#endif 
