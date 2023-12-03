#ifndef PANO_COMMON_HPP
#define PANO_COMMON_HPP

#include <cassert>
#include <chrono>
#include <opencv2/core/mat.hpp>
#include <options.hpp>
#include <vector>

// Serial operations used for benchmarking parallel speedup

std::vector<std::vector<double>> getSobelXKernel();
std::vector<std::vector<double>> getSobelYKernel();
std::vector<std::vector<double>> getGaussianKernel(int kernelSize,
                                                   double sigma);
cv::Mat convolveSequential(const cv::Mat &input,
                           const std::vector<std::vector<double>> kernel);
double computeSSDSequential(const cv::Mat &input1, const cv::Mat &input2);

// helper
cv::Mat __stitchAllSequential(std::vector<cv::Mat> images,
                              PanoramicOptions options);

cv::Mat timedStitchAllSequential(std::vector<cv::Mat> images,
                                 PanoramicOptions options);

std::vector<cv::KeyPoint>
seqHarrisCornerDetectorDetect(const cv::Mat &image,
                              HarrisCornerOptions options);

std::vector<cv::DMatch>
seqHarrisMatchKeyPoints(std::vector<cv::KeyPoint> keypointsL,
                        std::vector<cv::KeyPoint> keypointsR,
                        const cv::Mat &image1, const cv::Mat &image2,
                        const HarrisCornerOptions options, int offset);

inline void panic(const char *msg) {
  printf("%s\n", msg);
  exit(1);
}

class Timer {
public:
  Timer(const std::string &message, const bool shouldPrint)
      : message_(message), shouldPrint_(shouldPrint) {
    start_ = std::chrono::high_resolution_clock::now();
  }

  ~Timer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration<double, std::milli>(end - start_).count();
    if (shouldPrint_) {
      printf("%-40s %-10.2f ms\n", message_.c_str(), duration);
    }
  }

private:
  std::string message_;
  std::chrono::high_resolution_clock::time_point start_;
  bool shouldPrint_;
};

#endif
