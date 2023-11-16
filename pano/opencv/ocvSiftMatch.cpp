#include <matcher.hpp>
#include <opencv2/core/types.hpp>

std::vector<cv::DMatch>
OcvKeypointMatcher::matchKeyPoints(std::vector<cv::KeyPoint> keypointsL,
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
