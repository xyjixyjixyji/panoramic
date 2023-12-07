#include <common.hpp>
#include <detector.hpp>
#include <omp.h>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

OmpHarrisCornerDetector::OmpHarrisCornerDetector(HarrisCornerOptions options)
    : options_(options) {}

cv::Mat OmpConvolve(const cv::Mat &input,
                    const std::vector<std::vector<double>> kernel) {
  // kernel size has to be odd
  int kernelSize = kernel.size();
  assert(kernelSize % 2 == 1 && "Kernel size has to be odd");

  int k = kernelSize / 2;
  cv::Mat output(input.rows, input.cols, CV_64FC1);

#pragma omp parallel for collapse(2)
  for (int y = k; y < input.rows - k; y++) {
    for (int x = k; x < input.cols - k; x++) {
      double sum = 0.0;
      for (int i = -k; i <= k; i++) {
        for (int j = -k; j <= k; j++) {
          sum += input.at<double>(y + i, x + j) * kernel[k + i][k + j];
        }
      }

      output.at<double>(y, x) = sum;
    }
  }

  return output;
}

/**
 * @brief Detect keypoints in the input image by Harris corner method
 *
 * @param image input image is rather a grayscale image or a BGR image
 * @return std::vector<cv::KeyPoint> the keypoints detected
 */
std::vector<cv::KeyPoint>
OmpHarrisCornerDetector::detect(const cv::Mat &image) {
  // options
  const double k = options_.k_;
  const double thresh = options_.nmsThresh_;
  const double NMSNeighborhood = options_.nmsNeighborhood_;

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

  cv::Mat gradX = OmpConvolve(gray, sobelXKernel);
  cv::Mat gradY = OmpConvolve(gray, sobelYKernel);
  cv::Mat gradXX = gradX.mul(gradX);
  cv::Mat gradYY = gradY.mul(gradY);
  cv::Mat gradXY = gradX.mul(gradY);

  gradXX = OmpConvolve(gradXX, gaussianKernel);
  gradYY = OmpConvolve(gradYY, gaussianKernel);
  gradXY = OmpConvolve(gradXY, gaussianKernel);

  cv::Mat harrisResp = cv::Mat(gray.size(), CV_64F);

#pragma omp parallel for collapse(2)
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
  int halfLen = NMSNeighborhood / 2;

#pragma omp parallel shared(keypoints)
  {
    std::vector<cv::KeyPoint> localKeypoints;
#pragma omp for schedule(dynamic)
    for (int y = halfLen; y < gray.rows - halfLen; y++) {
      for (int x = halfLen; x < gray.cols - halfLen; x++) {
        double resp = harrisResp.at<double>(y, x);
        if (resp <= thresh)
          continue;

        double max_resp = std::numeric_limits<double>::min();
        for (int i = -halfLen; i <= halfLen; i++) {
          for (int j = -halfLen; j <= halfLen; j++) {
            if (i == 0 && j == 0)
              continue;
            max_resp = std::max(max_resp, harrisResp.at<double>(y + i, x + j));
            if (resp <= max_resp)
              goto bail;
          }
        }

        if (resp > max_resp) {
          localKeypoints.push_back(cv::KeyPoint(x, y, 1.f));
        }
      bail : {}
      }
    }

#pragma omp critical
    {
      keypoints.insert(keypoints.end(), localKeypoints.begin(),
                       localKeypoints.end());
    }
  }

  return keypoints;
}
