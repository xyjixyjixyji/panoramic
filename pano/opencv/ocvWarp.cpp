#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

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
