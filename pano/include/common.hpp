#ifndef PANO_COMMON_HPP
#define PANO_COMMON_HPP

#include <cassert>
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
double computeSSD(const cv::Mat &input1, const cv::Mat &input2);

// helper
cv::Mat stitchAllSequential(std::vector<cv::Mat> images,
                            PanoramicOptions options);

std::vector<cv::KeyPoint>
seqHarrisCornerDetectorDetect(const cv::Mat &image,
                              HarrisCornerOptions options);

std::vector<cv::DMatch>
seqHarrisMatchKeyPoints(std::vector<cv::KeyPoint> keypointsL,
                        std::vector<cv::KeyPoint> keypointsR,
                        const cv::Mat &image1, const cv::Mat &image2,
                        const HarrisCornerOptions options);

inline void panic(const char *msg) {
  printf("%s\n", msg);
  exit(1);
}

#endif
