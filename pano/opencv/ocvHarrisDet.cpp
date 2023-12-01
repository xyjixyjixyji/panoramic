#include <detector.hpp>
#include <opencv2/imgproc.hpp>

const float normalThresh = 0.05;

OcvHarrisCornerDetector::OcvHarrisCornerDetector(HarrisCornerOptions options)
    : options_(options) {}

std::vector<cv::KeyPoint>
OcvHarrisCornerDetector::detect(const cv::Mat &image) {
  std::vector<cv::KeyPoint> keypoints;

  // Convert to grayscale if necessary
  cv::Mat grayImage;
  if (image.channels() == 3) {
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
  } else {
    grayImage = image;
  }

  // Detecting Harris corners
  cv::Mat dst, dst_norm, dst_norm_scaled;
  dst = cv::Mat::zeros(grayImage.size(), CV_32FC1);
  cv::cornerHarris(grayImage, dst, options_.patchSize_, 3, options_.k_);

  // Normalizing
  cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
  cv::convertScaleAbs(dst_norm, dst_norm_scaled);

  // Drawing a circle around corners
  const int nmsRadius = options_.nmsNeighborhood_ / 2;
  for (int i = 0; i < dst_norm.rows; i++) {
    for (int j = 0; j < dst_norm.cols; j++) {
      if (rand() % 100 < 90)
        continue;

      float value = dst_norm.at<float>(i, j);
      if (value > normalThresh * 255) {
        // Check for the local maximum in the neighborhood
        bool localMax = true;
        for (int dx = -nmsRadius; dx <= nmsRadius; dx++) {
          for (int dy = -nmsRadius; dy <= nmsRadius; dy++) {
            int x = j + dx;
            int y = i + dy;
            if (x >= 0 && y >= 0 && x < dst_norm.cols && y < dst_norm.rows) {
              if (dst_norm.at<float>(y, x) > value) {
                localMax = false;
                break;
              }
            }
          }
          if (!localMax) {
            break;
          }
        }
        if (localMax) {
          keypoints.push_back(
              cv::KeyPoint(static_cast<float>(j), static_cast<float>(i), 1));
        }
      }
    }
  }

  return keypoints;
}
