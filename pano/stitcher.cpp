#include <common.hpp>
#include <stitcher.hpp>

cv::Mat Stitcher::stitch() {
  auto keypointsL = detector_->detect(imageL_);
  auto keypointsR = detector_->detect(imageR_);
  auto matches = matcher_->matchKeyPoints(keypointsL, keypointsR);

  // draw keypoints
  cv::Mat imgKeypointsL, imgKeypointsR;
  cv::drawKeypoints(imageL_, keypointsL, imgKeypointsL);
  cv::drawKeypoints(imageR_, keypointsR, imgKeypointsR);
  cv::imwrite("keypointsL.png", imgKeypointsL);
  cv::imwrite("keypointsR.png", imgKeypointsR);

  std::cout << "Keypoints in image 1: " << keypointsL.size() << std::endl;
  std::cout << "Keypoints in image 2: " << keypointsR.size() << std::endl;
  std::cout << "Found " << matches.size() << " matches" << std::endl;

  // draw matches
  cv::Mat imgMatches;
  cv::drawMatches(imageL_, keypointsL, imageR_, keypointsR, matches,
                  imgMatches);
  cv::imwrite("matches.png", imgMatches);

  // hMat maps points from image 1 to image 2
  cv::Mat hMat =
      homographyCalculator_->computeHomography(keypointsL, keypointsR, matches);
  cv::Mat warped = warpSequential(imageL_, imageR_, hMat);

  return warped;
}
