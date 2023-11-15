#include <common.hpp>
#include <stitcher.hpp>

cv::Mat Stitcher::stitch(const cv::Mat &imageL, const cv::Mat &imageR) {
  auto keypointsL = detector_->detect(imageL);
  auto keypointsR = detector_->detect(imageR);
  auto matches = matcher_->matchKeyPoints(keypointsL, keypointsR);
  // hMat maps points from image 1 to image 2
  cv::Mat hMat =
      homographyCalculator_->computeHomography(keypointsL, keypointsR, matches);
  cv::Mat warped = warp(imageL, imageR, hMat);

  return warped;
}
