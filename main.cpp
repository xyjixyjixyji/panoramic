#include <pano.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cassert>
#include <iostream>

int main(int argc, char **argv) {
  PanoramicOptions options = PanoramicOptions::getRuntimeOptions(argc, argv);
  std::vector<std::string> imgPaths = options.imgPaths_;
  assert(imgPaths.size() >= 2 && "Need at least 2 images to stitch");

  std::vector<cv::Mat> toWarped;
  for (auto &imgPath : imgPaths) {
    auto img = cv::imread(imgPath, cv::IMREAD_COLOR);
    toWarped.push_back(img);
  }

  cv::Mat warped = stitchAllSequential(toWarped, options);

  cv::imshow("Warped", warped);
  cv::waitKey(0);

  return 0;
}