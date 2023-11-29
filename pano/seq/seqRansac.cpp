#include <opencv2/calib3d.hpp>
#include <random>
#include <ransac.hpp>

SeqRansacHomographyCalculator::SeqRansacHomographyCalculator(
    RansacOptions options)
    : options_(options) {}

// H maps points from image 1 to image 2
cv::Mat SeqRansacHomographyCalculator::computeHomography(
    std::vector<cv::KeyPoint> &keypoints1,
    std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &matches) {
  // options
  const int numIterations = options_.numIterations_;
  const int numSamples = options_.numSamples_;
  const double distanceThreshold = options_.distanceThreshold_;

  // we random sample, and get the homography matrix with
  // the highest inlier count
  cv::Mat bestHomography;
  int bestInlierCount = 0;

  std::random_device rd;
  std::mt19937 rng(rd());

  for (int iter = 0; iter < numIterations; ++iter) {
    std::shuffle(matches.begin(), matches.end(), rng);
    std::vector<cv::Point2f> srcPoints, dstPoints;

    // Step 1: Randomly select a minimal subset of matches
    for (int j = 0; j < numSamples; j++) {
      srcPoints.push_back(keypoints1[matches[j].queryIdx].pt);
      dstPoints.push_back(keypoints2[matches[j].trainIdx].pt);
    }

    cv::Mat H = cv::findHomography(srcPoints, dstPoints);
    if (H.empty()) {
      continue;
    }

    int inlierCount = 0;
    for (const auto &match : matches) {
      cv::Point2f pt1 = keypoints1[match.queryIdx].pt;
      cv::Point2f pt2 = keypoints2[match.trainIdx].pt;
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
