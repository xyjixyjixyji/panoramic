#include <ransac.hpp>

OcvRansacHomographyCalculator::OcvRansacHomographyCalculator(
    RansacOptions options)
    : options_(options) {}

cv::Mat OcvRansacHomographyCalculator::computeHomography(
    std::vector<cv::KeyPoint> &keypoints1,
    std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &matches) {
  cv::Mat H;

  std::vector<cv::Point2f> points1, points2;
  for (auto &match : matches) {
    points1.push_back(keypoints1[match.queryIdx].pt);
    points2.push_back(keypoints2[match.trainIdx].pt);
  }

  H = cv::findHomography(points1, points2, cv::RANSAC,
                         options_.distanceThreshold_);

  return H;
}
