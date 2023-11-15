#include <detector.hpp>
#include <iostream>
#include <matcher.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

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

  cv::Mat img_matches;
  cv::drawMatches(imageL, keypoints1, imageR, keypoints2, matches, img_matches);

  // Show detected matches
  cv::imshow("Matches", img_matches);
  cv::waitKey(0); // Wait for a key press to exit

  return 0;
}