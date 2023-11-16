#include <cassert>
#include <common.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

std::vector<std::vector<double>> getSobelXKernel() {
  return {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
}

std::vector<std::vector<double>> getSobelYKernel() {
  return {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
}

std::vector<std::vector<double>> getGaussianKernel(int kernelSize,
                                                   double sigma) {
  std::vector<std::vector<double>> kernel(kernelSize,
                                          std::vector<double>(kernelSize));
  double sum = 0;
  for (int i = 0; i < kernelSize; ++i) {
    int x = i - kernelSize / 2;
    for (int j = 0; j < kernelSize; ++j) {
      int y = j - kernelSize / 2;
      kernel[i][j] = exp(-(x * x + y * y) / (2 * sigma * sigma));
      sum += kernel[i][j];
    }
  }
  for (auto &row : kernel) {
    for (auto &elem : row) {
      elem /= sum;
    }
  }
  return kernel;
}

/**
 * @brief Convolve the input image with the kernel
 *
 * @param input a grayscale image
 * @param kernel
 * @return cv::Mat output
 */
cv::Mat convolveSequential(const cv::Mat &input,
                           const std::vector<std::vector<double>> kernel) {
  // kernel size has to be odd
  int kernelSize = kernel.size();
  assert(kernelSize % 2 == 1 && "Kernel size has to be odd");

  int k = kernelSize / 2;
  cv::Mat output(input.rows, input.cols, CV_64FC1);

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

double computeSSD(const cv::Mat &input1, const cv::Mat &input2) {
  assert(input1.rows == input2.rows && input1.cols == input2.cols &&
         "Input images have to be of the same size");

  double sum = 0.0;
  for (int y = 0; y < input1.rows; y++) {
    for (int x = 0; x < input1.cols; x++) {
      double diff = input1.at<double>(y, x) - input2.at<double>(y, x);
      sum += diff * diff;
    }
  }

  return sum;
}

// Homography maps coordinates in imageL to imageR
cv::Mat warpSerial(cv::Mat imageL, cv::Mat imageR, cv::Mat homography) {
  cv::Mat warpedImage =
      cv::Mat::zeros(std::max(imageL.rows, imageR.rows),
                     imageL.cols + imageR.cols, imageL.type());

  // invH maps coordinates in imageR to imageL
  cv::Mat invH = homography.inv();

  // initialize the warped image
  warpedImage(cv::Rect(0, 0, imageL.cols, imageL.rows)) = imageL;
  warpedImage(cv::Rect(imageL.cols, 0, imageR.cols, imageR.rows)) = imageR;

  imageL.copyTo(warpedImage(cv::Rect(0, 0, imageL.cols, imageL.rows)));
  for (int y = 0; y < imageR.rows; y++) {
    for (int x = 0; x < imageR.cols; x++) {
      // we rotate the points in imageR into the imageL's space
      cv::Mat pt = (cv::Mat_<double>(3, 1) << x, y, 1.0);
      cv::Mat ptTransformed = invH * pt;
      ptTransformed /= ptTransformed.at<double>(2, 0);

      int newX = static_cast<int>(ptTransformed.at<double>(0, 0));
      int newY = static_cast<int>(ptTransformed.at<double>(1, 0));

      if (newX >= 0 && newX < warpedImage.cols && newY >= 0 &&
          newY < warpedImage.rows) {
        warpedImage.at<cv::Vec3b>(newY, newX) = imageR.at<cv::Vec3b>(y, x);
      }
    }
  }

  return warpedImage;
}
