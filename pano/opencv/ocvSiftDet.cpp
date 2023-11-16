#include <detector.hpp>
#include <opencv2/core/types.hpp>

std::vector<cv::KeyPoint> OcvSiftDetector::detect(const cv::Mat &image) {
  cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();
  std::vector<cv::KeyPoint> keypoints;
  sift->detect(image, keypoints);
  return keypoints;
}
