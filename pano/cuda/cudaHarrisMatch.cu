#include <common.hpp>
#include <limits>
#include <matcher.hpp>
#include <opencv2/core/types.hpp>
#include <sys/types.h>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(error));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__constant__ int cuPatchSize;
__constant__ uint64_t cuMaxSSDThreash;

CudaHarrisKeypointMatcher::CudaHarrisKeypointMatcher(
    cv::Mat &image1, cv::Mat &image2, HarrisCornerOptions options)
    : image1_(image1), image2_(image2), options_(options) {}

__global__ void matchKeypointsKernel(
    const float *kpsL_x, const float *kpsL_y, const float *kpsR_x,
    const float *kpsR_y, uchar3 *image1GPU, uchar3 *image2GPU,
    int numImage1Rows, int numImage1Cols, int numImage2Rows, int numImage2Cols,
    int numKpsL, int numKpsR, int *bestMatchIndices, uint64_t *bestMatchSSDs) {
  // Assuming each block processes one keypoint from keypointsL
  int keypointIdx = blockIdx.x;
  if (keypointIdx >= numKpsL)
    return;

  // Shared memory for the patch in image2GPU
  extern __shared__ uchar3 sharedPatch[];

  int patchSize = cuPatchSize;

  float pos1_x = kpsL_x[keypointIdx];
  float pos1_y = kpsL_y[keypointIdx];
  const int border = patchSize / 2;

  // Each thread in the block loads part of the patch into shared memory
  int lx = threadIdx.x;
  int ly = threadIdx.y;
  int globalX = pos1_x - border + lx;
  int globalY = pos1_y - border + ly;

  if (globalX >= 0 && globalX < numImage1Cols && globalY >= 0 &&
      globalY < numImage1Rows) {
    sharedPatch[ly * patchSize + lx] =
        image1GPU[globalY * numImage1Cols + globalX];
  }

  __syncthreads();

  // SSD calculation for the current keypoint against all keypoints in
  // keypointsR
  if (lx < patchSize && ly < patchSize) {
    int bestMatchIndex = -1;
    uint64_t bestMatchSSD = 0xffffffffffffffff;
    for (int j = 0; j < numKpsR; ++j) {
      float pos2_x = kpsR_x[j];
      float pos2_y = kpsR_y[j];

      // Compute SSD
      uint64_t ssd = 0;
      for (int dy = -border; dy <= border; ++dy) {
        for (int dx = -border; dx <= border; ++dx) {
          uchar3 p1 =
              sharedPatch[(ly + dy + border) * patchSize + (lx + dx + border)];
          int globalX2 = pos2_x + dx;
          int globalY2 = pos2_y + dy;
          uchar3 p2 = (globalX2 >= 0 && globalX2 < numImage2Cols &&
                       globalY2 >= 0 && globalY2 < numImage2Rows)
                          ? image2GPU[globalY2 * numImage2Cols + globalX2]
                          : make_uchar3(0, 0, 0);

          uint64_t diff = (p1.x - p2.x) * (p1.x - p2.x) +
                          (p1.y - p2.y) * (p1.y - p2.y) +
                          (p1.z - p2.z) * (p1.z - p2.z);
          ssd += diff;
        }
      }

      if (ssd < bestMatchSSD) {
        bestMatchSSD = ssd;
        bestMatchIndex = j;
      }
    }

    // Store the best match for this keypoint
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      if (bestMatchSSD < cuMaxSSDThreash) {
        bestMatchIndices[keypointIdx] = bestMatchIndex;
        bestMatchSSDs[keypointIdx] = bestMatchSSD;
      }
    }
  }
}

/**
 * @brief Match keypoints detected by Harris corner detector
 *
 * We typically take a patch around the keypoint and compare the distance
 *
 * @return std::vector<cv::DMatch> the matches
 */
std::vector<cv::DMatch> CudaHarrisKeypointMatcher::matchKeyPoints(
    std::vector<cv::KeyPoint> keypointsL,
    std::vector<cv::KeyPoint> keypointsR) {
  // options
  const int patchSize = options_.patchSize_;
  const uint64_t maxSSDThresh = options_.maxSSDThresh_;

  // copy to __constant__
  CUDA_CHECK(cudaMemcpyToSymbol(cuMaxSSDThreash, &maxSSDThresh,
                                sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyToSymbol(cuPatchSize, &patchSize, sizeof(int), 0,
                                cudaMemcpyHostToDevice));

  std::vector<cv::DMatch> matches;

  std::vector<float> pointsL_x(keypointsL.size()), pointsL_y(keypointsL.size());
  std::vector<float> pointsR_x(keypointsR.size()), pointsR_y(keypointsR.size());
  for (size_t i = 0; i < keypointsL.size(); ++i) {
    pointsL_x[i] = keypointsL[i].pt.x;
    pointsL_y[i] = keypointsL[i].pt.y;
  }
  for (size_t i = 0; i < keypointsR.size(); ++i) {
    pointsR_x[i] = keypointsR[i].pt.x;
    pointsR_y[i] = keypointsR[i].pt.y;
  }

  // memory allocation
  // Allocate memory for keypoints on GPU
  uchar3 *d_image1;
  uchar3 *d_image2;
  int numPixels = image1_.rows * image1_.cols;
  cudaMalloc(&d_image1, numPixels * sizeof(uchar3));
  cudaMalloc(&d_image2, numPixels * sizeof(uchar3));
  cudaMemcpy(d_image1, image1_.ptr(), numPixels * sizeof(uchar3),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_image2, image2_.ptr(), numPixels * sizeof(uchar3),
             cudaMemcpyHostToDevice);

  float *d_kpsL_X, *d_kpsL_Y, *d_kpsR_X, *d_kpsR_Y;
  CUDA_CHECK(cudaMalloc(&d_kpsL_X, keypointsL.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_kpsL_Y, keypointsL.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_kpsR_X, keypointsR.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_kpsR_Y, keypointsR.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_kpsL_X, pointsL_x.data(),
                        keypointsL.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kpsL_Y, pointsL_y.data(),
                        keypointsL.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kpsR_X, pointsR_x.data(),
                        keypointsR.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kpsR_Y, pointsR_y.data(),
                        keypointsR.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Allocate memory for the best match indices and SSDs on GPU
  int *d_bestMatchIndices;
  uint64_t *d_bestMatchSSDs;
  CUDA_CHECK(cudaMalloc(&d_bestMatchIndices, keypointsL.size() * sizeof(int)));
  CUDA_CHECK(
      cudaMalloc(&d_bestMatchSSDs, keypointsL.size() * sizeof(uint64_t)));

  // Initialize the best match arrays to -1 (or any sentinel value)
  CUDA_CHECK(
      cudaMemset(d_bestMatchIndices, -1, keypointsL.size() * sizeof(int)));
  CUDA_CHECK(
      cudaMemset(d_bestMatchSSDs, -1, keypointsL.size() * sizeof(double)));

  int numBlocks = keypointsL.size();
  dim3 blockSize(patchSize, patchSize);
  int sharedMemSize = patchSize * patchSize * sizeof(uchar3);
  matchKeypointsKernel<<<numBlocks, dim3(patchSize, patchSize),
                         sharedMemSize>>>(
      d_kpsL_X, d_kpsL_Y, d_kpsR_X, d_kpsR_Y, d_image1, d_image2, image1_.rows,
      image1_.cols, image2_.rows, image2_.cols, keypointsL.size(),
      keypointsR.size(), d_bestMatchIndices, d_bestMatchSSDs);
  cudaError_t kernelError = cudaGetLastError();
  if (kernelError != cudaSuccess) {
    printf("Kernel Error: %s\n", cudaGetErrorString(kernelError));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy the best match indices and SSDs back to the host
  std::vector<int> bestMatchIndices(keypointsL.size());
  std::vector<double> bestMatchSSDs(keypointsL.size());
  CUDA_CHECK(cudaMemcpy(bestMatchIndices.data(), d_bestMatchIndices,
                        keypointsL.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(bestMatchSSDs.data(), d_bestMatchSSDs,
                        keypointsL.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_kpsL_X));
  CUDA_CHECK(cudaFree(d_kpsL_Y));
  CUDA_CHECK(cudaFree(d_kpsR_X));
  CUDA_CHECK(cudaFree(d_kpsR_Y));
  CUDA_CHECK(cudaFree(d_bestMatchIndices));
  CUDA_CHECK(cudaFree(d_bestMatchSSDs));

  // get the matches
  for (size_t i = 0; i < keypointsL.size(); ++i) {
    if (bestMatchIndices[i] != -1) {
      matches.push_back(cv::DMatch(i, bestMatchIndices[i], bestMatchSSDs[i]));
    }
  }

  return matches;
}
