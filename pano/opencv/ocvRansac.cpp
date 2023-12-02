#include <opencv2/calib3d.hpp>
#include <ransac.hpp>

OcvRansacHomographyCalculator::OcvRansacHomographyCalculator(
    RansacOptions options)
    : options_(options) {}

cv::Mat OcvRansacHomographyCalculator::computeHomography(
    std::vector<cv::KeyPoint> &keypointsL,
    std::vector<cv::KeyPoint> &keypointsR, std::vector<cv::DMatch> &matches) {
  cv::Mat H;

  std::vector<cv::Point2f> ptsL, ptsR;
  for (auto &match : matches) {
    ptsL.push_back(keypointsL[match.queryIdx].pt);
    ptsR.push_back(keypointsR[match.trainIdx].pt);
  }

  H = cv::findHomography(ptsL, ptsR, cv::RANSAC, options_.distanceThreshold_,
                         cv::noArray(), options_.numIterations_, 0.995);

  return H;
}
