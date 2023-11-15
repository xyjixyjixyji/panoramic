#include <detector.hpp>
#include <matcher.hpp>
#include <ransac.hpp>
#include <stitcher.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

int main() {
  cv::Mat imageL = cv::imread("./data/viewL.png");
  cv::Mat imageR = cv::imread("./data/viewR.png");

  auto detector = SeqHarrisCornerDetector::createDetector();
  auto matcher = SeqHarrisKeyPointMatcher::createMatcher(imageL, imageR);
  auto ransac = SeqRansacHomographyCalculator::createHomographyCalculator();

  auto stitcher =
      Stitcher::createStitcher(detector, matcher, ransac, imageL, imageR);
  auto warped = stitcher->stitch(imageL, imageR);

  cv::imshow("Warped", warped);
  cv::waitKey(0);

  return 0;
}