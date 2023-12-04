#include <common.hpp>
#include <detector.hpp>

#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

CudaHarrisCornerDetector::CudaHarrisCornerDetector(HarrisCornerOptions options)
    : options_(options) {}

/**
 * @brief Detect keypoints in the input image by Harris corner method
 *
 * @param image input image is rather a grayscale image or a BGR image
 * @return std::vector<cv::KeyPoint> the keypoints detected
 */
std::vector<cv::KeyPoint>
CudaHarrisCornerDetector::detect(const cv::Mat &image) {
  return seqHarrisCornerDetectorDetect(image, options_);
}
