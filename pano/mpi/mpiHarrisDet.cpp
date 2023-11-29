#include <common.hpp>
#include <detector.hpp>
#include <mpi.h>
#include <opencv2/core/types.hpp>

// master pid
const int MASTER = 0;
// rows per chunk when master dispatches work
const int ROWS_PER_CHUNK = 10;
// tags
const int MPI_NUM_TAG = 0;
const int MPI_DATA_TAG = 1;
const int MPI_ROW_TAG = 2;
const int MPI_KILL_TAG = 3;

MPIHarrisCornerDetector::MPIHarrisCornerDetector(HarrisCornerOptions options,
                                                 int pid, int nproc)
    : options_(options), nproc_(nproc), pid_(pid) {}

void sendKeypoints(const std::vector<cv::KeyPoint> &keypoints, int dest,
                   MPI_Comm comm) {
  int numKeypoints = static_cast<int>(keypoints.size());
  MPI_Send(&numKeypoints, 1, MPI_INT, dest, MPI_NUM_TAG, comm);
  MPI_Send(keypoints.data(), numKeypoints * sizeof(cv::KeyPoint), MPI_BYTE,
           dest, MPI_DATA_TAG, comm);
}

std::vector<cv::KeyPoint> recvKeypoints(int source, MPI_Comm comm,
                                        MPI_Status &status) {
  MPI_Status _status;
  int numKeypoints;
  MPI_Recv(&numKeypoints, 1, MPI_INT, source, MPI_NUM_TAG, comm, &_status);

  std::vector<cv::KeyPoint> keypoints;
  keypoints.resize(numKeypoints);
  MPI_Recv(keypoints.data(), numKeypoints * sizeof(cv::KeyPoint), MPI_BYTE,
           _status.MPI_SOURCE, MPI_DATA_TAG, comm, &status);
  return keypoints;
}

std::vector<cv::KeyPoint>
MPIHarrisCornerDetector::detect(const cv::Mat &image) {
  std::vector<cv::KeyPoint> allKeypoints;

  if (pid_ == MASTER) {
    // Master Node
    int currentRow = 0;
    std::vector<MPI_Request> requests(nproc_ - 1);

    // Initially send work to all worker nodes
    for (int i = 1; i < nproc_; ++i) {
      int rowsToSend = (currentRow + ROWS_PER_CHUNK <= image.rows)
                           ? ROWS_PER_CHUNK
                           : image.rows - currentRow;
      if (rowsToSend > 0) {
        MPI_Send(&currentRow, 1, MPI_INT, i, MPI_ROW_TAG, MPI_COMM_WORLD);
        currentRow += rowsToSend;
      } else {
        MPI_Isend(nullptr, 0, MPI_BYTE, i, MPI_KILL_TAG, MPI_COMM_WORLD,
                  &requests[i - 1]);
      }
    }

    // Receive keypoints from workers and send new work if available
    int killCount = 0;
    while (true) {
      MPI_Status status;
      // Receive keypoints, status shows that the source is available
      std::vector<cv::KeyPoint> keypoints =
          recvKeypoints(MPI_ANY_SOURCE, MPI_COMM_WORLD, status);

      // Append keypoints to allKeypoints
      allKeypoints.insert(allKeypoints.end(), keypoints.begin(),
                          keypoints.end());

      // Send new work if available
      int rowsToSend = (currentRow + ROWS_PER_CHUNK <= image.rows)
                           ? ROWS_PER_CHUNK
                           : image.rows - currentRow;
      if (rowsToSend > 0) {
        MPI_Send(&currentRow, 1, MPI_INT, status.MPI_SOURCE, MPI_ROW_TAG,
                 MPI_COMM_WORLD);
        currentRow += rowsToSend;
      } else {
        MPI_Isend(nullptr, 0, MPI_BYTE, status.MPI_SOURCE, MPI_KILL_TAG,
                  MPI_COMM_WORLD, &requests[status.MPI_SOURCE - 1]);
        killCount++;
        if (killCount == nproc_ - 1) {
          break;
        }
      }
    }

    // Wait for all workers to complete their tasks
    MPI_Waitall(nproc_ - 1, requests.data(), MPI_STATUSES_IGNORE);
  } else {
    // Worker Nodes
    while (true) {
      MPI_Status status;
      int startRow;
      MPI_Recv(&startRow, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD,
               &status);

      if (status.MPI_TAG == MPI_KILL_TAG) {
        break; // No more work
      }

      int rowsToProcess = (startRow + ROWS_PER_CHUNK <= image.rows)
                              ? ROWS_PER_CHUNK
                              : image.rows - startRow;
      cv::Mat subImage = image(cv::Range(startRow, startRow + rowsToProcess),
                               cv::Range::all());

      // Perform Harris Corner Detection on the sub-image
      std::vector<cv::KeyPoint> localKeypoints = seqHarrisCornerDetectorDetect(
          subImage, options_); // Implement this function
      for (auto &keypoint : localKeypoints) {
        keypoint.pt.y += startRow;
      }

      // Send keypoints back to master (implement sendKeypoints function to
      // handle the data)
      sendKeypoints(localKeypoints, MASTER, MPI_COMM_WORLD);
    }
  }

  // broadcast the allKeypoints from MASTER
  int allKeypointsSize = allKeypoints.size();
  MPI_Bcast(&allKeypointsSize, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  if (pid_ != 0) {
    allKeypoints.resize(allKeypointsSize);
  }

  MPI_Bcast(&allKeypoints[0], allKeypoints.size() * sizeof(cv::KeyPoint),
            MPI_BYTE, MASTER, MPI_COMM_WORLD);

  return allKeypoints;
}
