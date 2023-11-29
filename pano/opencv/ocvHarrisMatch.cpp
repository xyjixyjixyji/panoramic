#include "matcher.hpp"

OcvHarrisKeypointMatcher::OcvHarrisKeypointMatcher(cv::Mat &imageL,
                                                   cv::Mat &imageR)
    : imageL_(imageL), imageR_(imageR) {}

std::vector<cv::DMatch>
OcvHarrisKeypointMatcher::matchKeyPoints(std::vector<cv::KeyPoint> keypointsL,
                                         std::vector<cv::KeyPoint> keypointsR) {
  cv::Mat descriptorsL, descriptorsR;
  cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();
  sift->compute(imageL_, keypointsL, descriptorsL);
  sift->compute(imageR_, keypointsR, descriptorsR);
  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  std::vector<cv::DMatch> matches;
  matcher->match(descriptorsL, descriptorsR, matches);
  return matches;
}
