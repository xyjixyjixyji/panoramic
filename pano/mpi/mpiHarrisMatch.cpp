#include <common.hpp>
#include <limits>
#include <matcher.hpp>
#include <mpi.h>
#include <numeric>
#include <opencv2/core/types.hpp>
#include <vector>

MPIHarrisKeypointMatcher::MPIHarrisKeypointMatcher(cv::Mat &image1,
                                                   cv::Mat &image2,
                                                   HarrisCornerOptions options,
                                                   int pid, int nproc)
    : image1_(image1), image2_(image2), options_(options), pid_(pid),
      nproc_(nproc) {}

/**
 * @brief Match keypoints detected by Harris corner detector
 *
 * We typically take a patch around the keypoint and compare the distance
 *
 * @return std::vector<cv::DMatch> the matches
 */
std::vector<cv::DMatch>
MPIHarrisKeypointMatcher::matchKeyPoints(std::vector<cv::KeyPoint> keypointsL,
                                         std::vector<cv::KeyPoint> keypointsR) {
  int totalKeypointsL = keypointsL.size();
  int keypointsPerProc = totalKeypointsL / nproc_;
  int startIdx = pid_ * keypointsPerProc;
  int endIdx =
      (pid_ == nproc_ - 1) ? totalKeypointsL : startIdx + keypointsPerProc;

  std::vector<cv::KeyPoint> localKeypointsL(keypointsL.begin() + startIdx,
                                            keypointsL.begin() + endIdx);

  // Perform local matching
  std::vector<cv::DMatch> localMatches = seqHarrisMatchKeyPoints(
      localKeypointsL, keypointsR, image1_, image2_, options_);

  // Gather the number of matches from each process
  std::vector<int> matchCounts(nproc_);
  int localMatchCount = static_cast<int>(localMatches.size());
  MPI_Allgather(&localMatchCount, 1, MPI_INT, matchCounts.data(), 1, MPI_INT,
                MPI_COMM_WORLD);

  // Calculate total matches and displacements for gathering
  int totalMatches = std::accumulate(matchCounts.begin(), matchCounts.end(), 0);

  // Calculate displacements in bytes
  std::vector<int> byteCounts(nproc_);
  for (int i = 0; i < nproc_; ++i) {
    byteCounts[i] = matchCounts[i] * sizeof(cv::DMatch);
  }
  std::vector<int> byteDispl(nproc_);
  byteDispl[0] = 0;
  for (int i = 1; i < nproc_; ++i) {
    byteDispl[i] = byteDispl[i - 1] + byteCounts[i - 1];
  }

  // Gather all matches
  std::vector<cv::DMatch> allMatches(totalMatches);
  MPI_Allgatherv(localMatches.data(), localMatchCount * sizeof(cv::DMatch),
                 MPI_BYTE, allMatches.data(), byteCounts.data(),
                 byteDispl.data(), MPI_BYTE, MPI_COMM_WORLD);

  return allMatches;
}
