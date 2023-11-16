#include <common.hpp>
#include <stitcher.hpp>

cv::Mat Stitcher::stitch() {
  auto keypointsL = detector_->detect(imageL_);
  auto keypointsR = detector_->detect(imageR_);
  auto matches = matcher_->matchKeyPoints(keypointsL, keypointsR);

  std::cout << "Keypoints in image 1: " << keypointsL.size() << std::endl;
  std::cout << "Keypoints in image 2: " << keypointsR.size() << std::endl;
  std::cout << "Found " << matches.size() << " matches" << std::endl;

  // hMat maps points from image 1 to image 2
  cv::Mat hMat =
      homographyCalculator_->computeHomography(keypointsL, keypointsR, matches);
  cv::Mat warped = warpSequential(imageL_, imageR_, hMat);

  return warped;
}
