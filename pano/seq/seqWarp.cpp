#include "warp.hpp"

// Homography maps coordinates in imageL to imageR
cv::Mat warpSequential(cv::Mat imageL, cv::Mat imageR, cv::Mat homography) {
  cv::Mat warpedImage =
      cv::Mat::zeros(std::max(imageL.rows, imageR.rows),
                     imageL.cols + imageR.cols, imageL.type());

  // invH maps coordinates in imageR to imageL
  cv::Mat invH = homography.inv();

  // initialize the warped image
  warpedImage(cv::Rect(0, 0, imageL.cols, imageL.rows)) = imageL;
  warpedImage(cv::Rect(imageL.cols, 0, imageR.cols, imageR.rows)) = imageR;

  imageL.copyTo(warpedImage(cv::Rect(0, 0, imageL.cols, imageL.rows)));
  for (int y = 0; y < imageR.rows; y++) {
    for (int x = 0; x < imageR.cols; x++) {
      // we rotate the points in imageR into the imageL's space
      cv::Mat pt = (cv::Mat_<double>(3, 1) << x, y, 1.0);
      cv::Mat ptTransformed = invH * pt;
      ptTransformed /= ptTransformed.at<double>(2, 0);

      int newX = static_cast<int>(ptTransformed.at<double>(0, 0));
      int newY = static_cast<int>(ptTransformed.at<double>(1, 0));

      if (newX >= 0 && newX < warpedImage.cols && newY >= 0 &&
          newY < warpedImage.rows) {
        warpedImage.at<cv::Vec3b>(newY, newX) = imageR.at<cv::Vec3b>(y, x);
      }
    }
  }

  return warpedImage;
}
