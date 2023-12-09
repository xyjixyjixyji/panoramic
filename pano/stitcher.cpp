#include <common.hpp>
#include <options.hpp>
#include <stitcher.hpp>
#include <iostream>
#include <fstream>

cv::Mat warpOcv(cv::Mat imageL, cv::Mat imageR, cv::Mat homography) {
  // Create an output image large enough to hold both images
  cv::Mat invH = homography.inv();
  cv::Mat warpedImage =
      cv::Mat::zeros(std::max(imageL.rows, imageR.rows),
                     imageL.cols + imageR.cols, imageL.type());

  // Place imageL in the left part of the output image
  imageL.copyTo(warpedImage(cv::Rect(0, 0, imageL.cols, imageL.rows)));

  // Apply the modified homography transformation to imageR
  cv::warpPerspective(imageR, warpedImage, invH, warpedImage.size(),
                      cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);

  return warpedImage;
}

//! NORMALLY DO NOT EDIT, this is a little bit messy since we are timing it
cv::Mat Stitcher::stitch() {
  bool shouldPrint =
      !options_.use_mpi_ || (options_.use_mpi_ && options_.pid_ == 0);

  if (shouldPrint) {
    printf("========== %-20s ==========\n", "Stitching");
  }

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
    warped = warpOcv(imageL_, imageR_, hMat);
  }

  if (shouldPrint) {
    printf("%ld keypoints detected in image L\n", keypointsL.size());
    printf("%ld keypoints detected in image R\n", keypointsR.size());
    printf("%ld matches found\n", matches.size());
    printf("========== %-20s ==========\n", "Done");
  }

  return warped;
}
