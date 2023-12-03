#include <opencv2/calib3d.hpp>
#include <random>
#include <ransac.hpp>

OmpRansacHomographyCalculator::OmpRansacHomographyCalculator(
    RansacOptions options)
    : options_(options) {}

// H maps points from image 1 to image 2
cv::Mat OmpRansacHomographyCalculator::computeHomography(
    std::vector<cv::KeyPoint> &keypoints1,
    std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &matches) {
  // options
  const int numIterations = options_.numIterations_;
  const int numSamples = options_.numSamples_;
  const double distanceThreshold = options_.distanceThreshold_;

  cv::Mat bestHomography;
  int bestInlierCount = 0;

#pragma omp parallel
  {
    // Separate random generator for each thread to avoid contention
    std::random_device rd;
    std::mt19937 rng(rd());

    cv::Mat localBestHomography;
    int localBestInlierCount = 0;

#pragma omp for nowait
    for (int iter = 0; iter < numIterations; ++iter) {
      std::vector<cv::DMatch> localMatches =
          matches; // Copy of matches for each thread
      std::shuffle(localMatches.begin(), localMatches.end(), rng);
      std::vector<cv::Point2f> srcPoints, dstPoints;

      for (int j = 0; j < numSamples; j++) {
        srcPoints.push_back(keypoints1[localMatches[j].queryIdx].pt);
        dstPoints.push_back(keypoints2[localMatches[j].trainIdx].pt);
      }

      cv::Mat H = cv::findHomography(srcPoints, dstPoints);
      if (H.empty()) {
        continue;
      }

      int inlierCount = 0;
      for (const auto &match : localMatches) {
        cv::Point2f pt1 = keypoints1[match.queryIdx].pt;
        cv::Point2f pt2 = keypoints2[match.trainIdx].pt;
        cv::Mat pt1Mat = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1.0);
        cv::Mat pt2Transformed = H * pt1Mat;
        pt2Transformed /= pt2Transformed.at<double>(2, 0);
        cv::Point2f pt2Estimate(pt2Transformed.at<double>(0, 0),
                                pt2Transformed.at<double>(1, 0));

        if (cv::norm(pt2Estimate - pt2) < distanceThreshold)
          inlierCount++;
      }

      // Update best homography in a thread-safe manner
      if (inlierCount > localBestInlierCount) {
        localBestInlierCount = inlierCount;
        localBestHomography = H;
      }
    }

    // Critical section for updating the global bestHomography
    if (localBestInlierCount > bestInlierCount) {
#pragma omp critical
      {
        bestInlierCount = localBestInlierCount;
        bestHomography = localBestHomography;
      }
    }
  }

  return bestHomography;
}
