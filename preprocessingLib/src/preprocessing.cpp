#include "preprocessingLib.h"

namespace preprocessingLib {
    cv::Mat imread(const std::string& filename, int flags) {
        return cv::imread(filename, flags);
    }
}
