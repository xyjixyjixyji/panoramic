#ifndef PANO_WARP_HPP
#define PANO_WARP_HPP

#include <opencv2/opencv.hpp>

typedef std::function<cv::Mat(cv::Mat, cv::Mat, cv::Mat)> warpFunction_t;

cv::Mat warpSequential(cv::Mat imageL, cv::Mat imageR, cv::Mat homography);

#endif
