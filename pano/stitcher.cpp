#include "options.hpp"
#include <chrono>
#include <common.hpp>
#include <stitcher.hpp>

//! NORMALLY DO NOT EDIT, this is a little bit messy since we are timing it
cv::Mat Stitcher::stitch() {
  bool shouldPrint =
      !options_.use_mpi_ || (options_.use_mpi_ && options_.pid_ == 0);
  if (shouldPrint) {
    printf("========== %-20s ==========\n", "Stitching");


  Timer e2eTimer("End to end time to stitch", shouldPrint);

  std::vector<cv::KeyPoint> keypointsL;
  {
    Timer timer("Time to detect first keypoints", shouldPrint);
    keypointsL = detector_->detect(imageL_);
  }

  std::vector<cv::KeyPoint> keypointsR;
  {
    Timer timer("Time to detect second keypoints", shouldPrint);
    keypointsR = detector_->detect(imageR_);
  }

  std::vector<cv::DMatch> matches;
  {
    Timer timer("Time to match keypoints", shouldPrint);
    matches = matcher_->matchKeyPoints(keypointsL, keypointsR);
  }

  cv::Mat hMat;
  {
    Timer timer("Time to compute homography", shouldPrint);
    hMat = homographyCalculator_->computeHomography(keypointsL, keypointsR,
                                                    matches);
  }

  cv::Mat warped;
  {
    Timer timer("Time to warp image", shouldPrint);
    warped = warpFunction_(imageL_, imageR_, hMat);
  }

  if (shouldPrint) {
    printf("========== %-20s ==========\n", "Done");

  return warped;
}
