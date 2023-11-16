#include <detector.hpp>
#include <matcher.hpp>
#include <options.hpp>
#include <ransac.hpp>
#include <stitcher.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

int main(int argc, char **argv) {
  PanoramicOptions options = PanoramicOptions::getRuntimeOptions(argc, argv);
  cv::Mat imageL = cv::imread(options.imgLPath_);
  cv::Mat imageR = cv::imread(options.imgRPath_);

  auto stitcher = Stitcher::createStitcher(imageL, imageR, options);
  auto warped = stitcher->stitch();

  cv::imshow("Warped", warped);
  cv::waitKey(0);

  return 0;
}