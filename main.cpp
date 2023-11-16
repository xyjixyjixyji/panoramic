#include <detector.hpp>
#include <matcher.hpp>
#include <options.hpp>
#include <ransac.hpp>
#include <stitcher.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

int main(int argc, char **argv) {
  cv::Mat imageL = cv::imread("./data/viewL.png");
  cv::Mat imageR = cv::imread("./data/viewR.png");

  PanoramicOptions options = PanoramicOptions::getRuntimeOptions(argc, argv);

  auto stitcher = Stitcher::createStitcher(imageL, imageR, options);
  auto warped = stitcher->stitch(imageL, imageR);

  cv::imshow("Warped", warped);
  cv::waitKey(0);

  return 0;
}