#ifndef PANO_COMMON_HPP
#define PANO_COMMON_HPP

#include <cassert>
#include <opencv2/core/mat.hpp>
#include <vector>

// Serial operations used for benchmarking parallel speedup

std::vector<std::vector<double>> getSobelXKernel();
std::vector<std::vector<double>> getSobelYKernel();
std::vector<std::vector<double>> getGaussianKernel(int kernelSize,
                                                   double sigma);
cv::Mat convolveSequential(const cv::Mat &input,
                           const std::vector<std::vector<double>> kernel);
double computeSSD(const cv::Mat &input1, const cv::Mat &input2);
cv::Mat warpSerial(cv::Mat imageL, cv::Mat imageR, cv::Mat homography);

#endif
