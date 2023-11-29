#include <detector.hpp>
#include <opencv2/imgproc.hpp>

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

  cv::Mat dst, dst_norm, dst_norm_scaled;
  dst = cv::Mat::zeros(image.size(), CV_32FC1);

  // Detecting corners
  cv::cornerHarris(grayImage, dst, 3, 3, options_.k_);

  // Normalizing
  cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1);
  cv::convertScaleAbs(dst_norm, dst_norm_scaled);

  // Locating and adding keypoints
  double thresh = 0.15; // Threshold for detecting corners
  int nmsBlockSize = options_.nmsNeighborhood_;
  for (int i = nmsBlockSize; i < dst_norm.rows - nmsBlockSize; i++) {
    for (int j = nmsBlockSize; j < dst_norm.cols - nmsBlockSize; j++) {
      if (dst_norm.at<float>(i, j) > thresh * 255) {
        bool isLocalMax = true;

        // Check if this corner is the local maximum in its neighborhood
        for (int dy = -nmsBlockSize; dy <= nmsBlockSize; dy++) {
          for (int dx = -nmsBlockSize; dx <= nmsBlockSize; dx++) {
            if (dst_norm.at<float>(i + dy, j + dx) > dst_norm.at<float>(i, j)) {
              isLocalMax = false;
              break;
            }
          }
          if (!isLocalMax)
            break;
        }

        if (isLocalMax) {
          keypoints.push_back(cv::KeyPoint(float(j), float(i), 1.f));
        }
      }
    }
  }

  return keypoints;
}
