#include <cassert>
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
  std::vector<std::string> imgPaths = options.imgPaths_;
  assert(imgPaths.size() >= 2 && "Need at least 2 images to stitch");

  cv::Mat imageL = cv::imread(imgPaths[0]);
  cv::Mat imageR = cv::imread(imgPaths[1]);

  auto stitcher = Stitcher::createStitcher(imageL, imageR, options);
  auto warped = stitcher->stitch();

  cv::imshow("Warped", warped);
  cv::waitKey(0);

  return 0;
}