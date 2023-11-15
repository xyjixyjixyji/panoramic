#include <opencv2/calib3d.hpp>
#include <random>
#include <ransac.hpp>

const int numIterations = 500;
// # of samples we are using for each RANSAC iteration
const int numSamples = 4;
// the distance that we are tolerating for a point to be considered an inlier
const double distanceThreshold = 5.0;

// H maps points from image 1 to image 2
cv::Mat SeqRansacHomographyCalculator::computeHomography() {
  // we random sample, and get the homography matrix with
  // the highest inlier count
  cv::Mat bestHomography;
  int bestInlierCount = 0;

  std::random_device rd;
  std::mt19937 rng(rd());

  for (int iter = 0; iter < numIterations; ++iter) {
    std::shuffle(matches_.begin(), matches_.end(), rng);
    std::vector<cv::Point2f> srcPoints, dstPoints;

    // Step 1: Randomly select a minimal subset of matches
    for (int j = 0; j < numSamples; j++) {
      srcPoints.push_back(keypoints1_[matches_[j].queryIdx].pt);
      dstPoints.push_back(keypoints2_[matches_[j].trainIdx].pt);
    }

    cv::Mat H = cv::findHomography(srcPoints, dstPoints);

    int inlierCount = 0;
    for (const auto &match : matches_) {
      cv::Point2f pt1 = keypoints1_[match.queryIdx].pt;
      cv::Point2f pt2 = keypoints2_[match.trainIdx].pt;
      cv::Mat pt1Mat = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1.0);

      // generate the estimate of pt2
      cv::Mat pt2Transformed = H * pt1Mat;
      pt2Transformed /= pt2Transformed.at<double>(2, 0); // Normalize
      cv::Point2f pt2Estimate(pt2Transformed.at<double>(0, 0),
                              pt2Transformed.at<double>(1, 0));

      if (cv::norm(pt2Estimate - pt2) < distanceThreshold)
        inlierCount++;
    }

    if (inlierCount > bestInlierCount) {
      bestInlierCount = inlierCount;
      bestHomography = H;
    }
  }

  return bestHomography;
}