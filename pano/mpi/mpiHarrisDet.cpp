#include <common.hpp>
#include <detector.hpp>
#include <mpi.h>

MPIHarrisCornerDetector::MPIHarrisCornerDetector(HarrisCornerOptions options,
                                                 int pid, int nproc)
    : options_(options), nproc_(nproc), pid_(pid) {}

// FIXME: we can certainly avoid the Allgather and Allgatherv
//        by letting each thread holds a vector of keypoints
//        and match them with the keypoints from other threads
std::vector<cv::KeyPoint>
MPIHarrisCornerDetector::detect(const cv::Mat &image) {
  int rows_per_process = image.rows / nproc_;
  int start_row = pid_ * rows_per_process;
  int end_row =
      (pid_ == nproc_ - 1) ? image.rows : start_row + rows_per_process;

  // Create a sub-image for this process
  cv::Mat subImage = image(cv::Range(start_row, end_row), cv::Range::all());

  // Perform Harris Corner Detection on the sub-image
  std::vector<cv::KeyPoint> localKeypoints =
      seqHarrisCornerDetectorDetect(subImage, options_);

  // offset the row
  for (auto &keypoint : localKeypoints) {
    keypoint.pt.y += start_row;
  }

  // Gather the number of keypoints detected by each process
  std::vector<int> keypointNums(nproc_);
  int localNumKeypoints = static_cast<int>(localKeypoints.size());
  MPI_Allgather(&localNumKeypoints, 1, MPI_INT, keypointNums.data(), 1, MPI_INT,
                MPI_COMM_WORLD);

  // Calculate the total number of keypoints and resize allKeypoints vector in
  // all processes
  int totalKeypoints =
      std::accumulate(keypointNums.begin(), keypointNums.end(), 0);
  std::vector<cv::KeyPoint> allKeypoints(totalKeypoints);

  // Calculate displacements in bytes
  std::vector<int> byteCounts(nproc_);
  for (int i = 0; i < nproc_; ++i) {
    byteCounts[i] = keypointNums[i] * sizeof(cv::KeyPoint);
  }
  std::vector<int> byteDispl(nproc_);
  byteDispl[0] = 0;
  for (int i = 1; i < nproc_; ++i) {
    byteDispl[i] = byteDispl[i - 1] + byteCounts[i - 1];
  }

  // Use MPI_Allgatherv to gather keypoints from all processes
  MPI_Allgatherv(localKeypoints.data(),
                 localNumKeypoints * sizeof(cv::KeyPoint), MPI_BYTE,
                 allKeypoints.data(), byteCounts.data(), byteDispl.data(),
                 MPI_BYTE, MPI_COMM_WORLD);

  return allKeypoints;
}
