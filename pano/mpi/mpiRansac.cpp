#include <mpi.h>
#include <opencv2/calib3d.hpp>
#include <random>
#include <ransac.hpp>

MPIRansacHomographyCalculator::MPIRansacHomographyCalculator(
    RansacOptions options, int pid, int nproc)
    : options_(options), pid_(pid), nproc_(nproc) {}

// H maps points from image 1 to image 2
cv::Mat MPIRansacHomographyCalculator::computeHomography(
    std::vector<cv::KeyPoint> &keypoints1,
    std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &matches) {
  // options
  const int numIterations = options_.numIterations_;
  const int numSamples = options_.numSamples_;
  const double distanceThreshold = options_.distanceThreshold_;

  // we random sample, and get the homography matrix with
  // the highest inlier count
  cv::Mat bestHomography = cv::Mat::zeros(3, 3, CV_64F);
  int bestInlierCount = 0;

  std::random_device rd;
  std::mt19937 rng(rd());

  size_t matchesPerNode = matches.size() / nproc_;
  size_t start = pid_ * matchesPerNode;
  size_t end = (pid_ == nproc_ - 1) ? matches.size() : start + matchesPerNode;

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

    int localInlierCount = 0;
    // each MPI node only checks a subset of the matches
    for (size_t i = start; i < end; i++) {
      auto match = matches[i];
      cv::Point2f pt1 = keypoints1[match.queryIdx].pt;
      cv::Point2f pt2 = keypoints2[match.trainIdx].pt;
      cv::Mat pt1Mat = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1.0);

      // generate the estimate of pt2
      cv::Mat pt2Transformed = H * pt1Mat;
      pt2Transformed /= pt2Transformed.at<double>(2, 0); // Normalize
      cv::Point2f pt2Estimate(pt2Transformed.at<double>(0, 0),
                              pt2Transformed.at<double>(1, 0));

      if (cv::norm(pt2Estimate - pt2) < distanceThreshold)
        localInlierCount++;
    }

    // each mpi node aggregates their inlier count
    int totalInlierCount;
    MPI_Reduce(&localInlierCount, &totalInlierCount, 1, MPI_INT, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (pid_ == 0 && totalInlierCount > bestInlierCount) {
      bestInlierCount = totalInlierCount;
      bestHomography = H;
    }
  }

  MPI_Bcast(bestHomography.data, 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return bestHomography;
}
