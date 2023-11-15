#include <detector.hpp>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

int main() {
  cv::Mat image = cv::imread("./data/waffle.png");
  auto det = SeqHarrisCornerDetector::createDetector();
  auto keypoints = det->detect(image);

  std::cout << "Number of keypoints detected: " << keypoints.size() << "\n";

  // draw the keypoints
  cv::Mat outImage(image.size(), CV_8UC3);
  cv::drawKeypoints(image, keypoints, outImage);

  cv::imshow("Keypoints", outImage);
  cv::waitKey(0);

  return 0;
}