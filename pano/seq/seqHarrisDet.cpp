#include <common.hpp>
#include <detector.hpp>

#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

// Smaller k leads to more sensitive detection
// Empirically, k is in [0.04, 0.06]
const double k = 0.04;

// threshold for non-maximum suppression, a higher threshold will lead to less
// keypoints
const double thresh = 300000;

// WARN: has to be odd
// the neighbor hood is [x-NMSNeighborhood, x+NMSNeighborhood]
const double NMSNeighborhood = 9;

/**
 * @brief Detect keypoints in the input image by Harris corner method
 *
 * @param image input image is rather a grayscale image or a BGR image
 * @return std::vector<cv::KeyPoint> the keypoints detected
 */
std::vector<cv::KeyPoint>
SeqHarrisCornerDetector::detect(const cv::Mat &image) {
  std::vector<cv::KeyPoint> keypoints;
  auto sobelXKernel = getSobelXKernel();
  auto sobelYKernel = getSobelYKernel();
  auto gaussianKernel = getGaussianKernel(5, 1.0);

  // Ensure the image is grayscale
  cv::Mat gray;
  if (image.channels() == 3) {
    cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = image;
  }
  gray.convertTo(gray, CV_64F); // Ensure the image is in double format

  cv::Mat gradX = convolveSequential(gray, sobelXKernel);
  cv::Mat gradY = convolveSequential(gray, sobelYKernel);
  cv::Mat gradXX = gradX.mul(gradX);
  cv::Mat gradYY = gradY.mul(gradY);
  cv::Mat gradXY = gradX.mul(gradY);

  gradXX = convolveSequential(gradXX, gaussianKernel);
  gradYY = convolveSequential(gradYY, gaussianKernel);
  gradXY = convolveSequential(gradXY, gaussianKernel);

  cv::Mat harrisResp = cv::Mat(gray.size(), CV_64F);
  for (int y = 0; y < gray.rows; y++) {
    for (int x = 0; x < gray.cols; x++) {
      double xx = gradXX.at<double>(y, x);
      double yy = gradYY.at<double>(y, x);
      double xy = gradXY.at<double>(y, x);
      double det = xx * yy - xy * xy;
      double trace = xx + yy;
      harrisResp.at<double>(y, x) = det - k * trace * trace;
    }
  }

  // Non-maximum suppression
  for (int y = 1; y < gray.rows; y++) {
    for (int x = 1; x < gray.cols; x++) {
      double resp = harrisResp.at<double>(y, x);
      if (resp <= thresh)
        continue;

      // find the max around this point
      double max_resp = std::numeric_limits<double>::min();
      int k = NMSNeighborhood / 2;
      for (int i = -k; i <= k; i++) {
        for (int j = -k; j <= k; j++) {
          if (i == 0 && j == 0)
            continue;
          max_resp = std::max(max_resp, harrisResp.at<double>(y + i, x + j));
        }
      }

      if (resp > max_resp) {
        keypoints.push_back(cv::KeyPoint(x, y, 1.f));
      }
    }
  }

  return keypoints;
}