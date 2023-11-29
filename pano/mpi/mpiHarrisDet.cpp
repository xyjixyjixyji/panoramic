#include <common.hpp>
#include <detector.hpp>
#include <mpi.h>

MPIHarrisCornerDetector::MPIHarrisCornerDetector(HarrisCornerOptions options,
                                                 int pid, int nproc)
    : options_(options), nproc_(nproc), pid_(pid) {}

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

  // Calculate displacements for MPI_Allgatherv
  std::vector<int> displ(nproc_);
  displ[0] = 0;
  for (int i = 1; i < nproc_; ++i) {
    displ[i] = displ[i - 1] + keypointNums[i - 1];
  }

  // Use MPI_Allgatherv to gather keypoints from all processes
  MPI_Allgatherv(localKeypoints.data(),
                 localNumKeypoints * sizeof(cv::KeyPoint), MPI_BYTE,
                 allKeypoints.data(), keypointNums.data(), displ.data(),
                 MPI_BYTE, MPI_COMM_WORLD);

  return allKeypoints;

  // TODO: prevent all gatherv
  // // Gather the number of keypoints detected by each process
  // std::vector<int> keypointNums(nproc_);
  // int localNumKeypoints = static_cast<int>(localKeypoints.size());
  // MPI_Gather(&localNumKeypoints, 1, MPI_INT, keypointNums.data(), 1, MPI_INT,
  // 0,
  //            MPI_COMM_WORLD);

  // // Gather all keypoints from all processes
  // std::vector<cv::KeyPoint> allKeypoints;
  // if (pid_ == 0) {
  //   // Only the root process needs to allocate space for all keypoints
  //   int totalKeypoints =
  //       std::accumulate(keypointNums.begin(), keypointNums.end(), 0);
  //   allKeypoints.resize(totalKeypoints);
  // }

  // // calculate displacement
  // std::vector<int> displ(nproc_);
  // displ[0] = 0; // The first displacement is always 0
  // for (int i = 1; i < nproc_; ++i) {
  //   displ[i] = displ[i - 1] + keypointNums[i - 1];
  // }

  // MPI_Gatherv(localKeypoints.data(), localNumKeypoints *
  // sizeof(cv::KeyPoint),
  //             MPI_BYTE, allKeypoints.data(), keypointNums.data(),
  //             displ.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

  // return allKeypoints;
}
