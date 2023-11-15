#include <detector.hpp>
#include <iostream>
#include <matcher.hpp>
#include <ransac.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

cv::Mat warp(cv::Mat imageL, cv::Mat imageR, cv::Mat homography) {
  cv::Mat warpedImage(std::max(imageL.rows, imageR.rows),
                      imageL.cols + imageR.cols, imageL.type());

  cv::Mat invH = homography.inv();

  // initialize the warped image
  warpedImage(cv::Rect(0, 0, imageL.cols, imageL.rows)) = imageL;
  warpedImage(cv::Rect(imageL.cols, 0, imageR.cols, imageR.rows)) = imageR;

  imageL.copyTo(warpedImage(cv::Rect(0, 0, imageL.cols, imageL.rows)));
  for (int y = 0; y < imageR.rows; y++) {
    for (int x = 0; x < imageR.cols; x++) {
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

int main() {
  cv::Mat imageL = cv::imread("./data/viewL.png");
  cv::Mat imageR = cv::imread("./data/viewR.png");
  auto det = SeqHarrisCornerDetector::createDetector();
  auto keypoints1 = det->detect(imageL);
  auto keypoints2 = det->detect(imageR);

  std::cout << "Number of keypoints detected: " << keypoints1.size() << "\n";
  std::cout << "Number of keypoints detected: " << keypoints2.size() << "\n";

  auto matcher = SeqHarrisKeyPointMatcher::createMatcher(
      imageL, imageR, keypoints1, keypoints2);

  auto matches = matcher->matchKeyPoints();

  std::cout << "Number of matches: " << matches.size() << "\n";

  auto ransac = SeqRansacHomographyCalculator::createHomographyCalculator(
      keypoints1, keypoints2, matches);
  // hMat maps points from image 1 to image 2
  cv::Mat hMat = ransac->computeHomography();

  cv::Mat warped = warp(imageL, imageR, hMat);

  cv::imshow("Warped", warped);
  cv::waitKey(0);

  return 0;
}