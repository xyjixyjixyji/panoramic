#include <common.hpp>
#include <detector.hpp>

#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

OmpHarrisCornerDetector::OmpHarrisCornerDetector(HarrisCornerOptions options, int nproc)
    : options_(options), nproc_(nproc) {}

/**
 * @brief Detect keypoints in the input image by Harris corner method
 *
 * @param image input image is rather a grayscale image or a BGR image
 * @return std::vector<cv::KeyPoint> the keypoints detected
 */
std::vector<cv::KeyPoint>
OmpHarrisCornerDetector::detect(const cv::Mat &image) {
    // TODO: 
    (void) nproc_;
    return seqHarrisCornerDetectorDetect(image, options_);
}
