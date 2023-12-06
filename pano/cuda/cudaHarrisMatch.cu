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

texture<uchar4, 2, cudaReadModeElementType> texImage1;
texture<uchar4, 2, cudaReadModeElementType> texImage2;

void setupTexture(cudaArray *cuArray1, cudaArray *cuArray2) {
  texImage1.addressMode[0] = cudaAddressModeWrap;
  texImage1.addressMode[1] = cudaAddressModeWrap;
  texImage1.filterMode = cudaFilterModePoint;
  texImage1.normalized = false;

  texImage2.addressMode[0] = cudaAddressModeWrap;
  texImage2.addressMode[1] = cudaAddressModeWrap;
  texImage2.filterMode = cudaFilterModePoint;
  texImage2.normalized = false;

  cudaBindTextureToArray(texImage1, cuArray1);
  cudaBindTextureToArray(texImage2, cuArray2);
}

CudaHarrisKeypointMatcher::CudaHarrisKeypointMatcher(
    cv::Mat &image1, cv::Mat &image2, HarrisCornerOptions options)
    : image1_(image1), image2_(image2), options_(options) {
  auto cudaChennelDesc = cudaCreateChannelDesc<uchar4>();
  cudaArray *cuArray1, *cuArray2;
  cudaMallocArray(&cuArray1, &cudaChennelDesc, image1_.cols, image1_.rows);
  cudaMallocArray(&cuArray2, &cudaChennelDesc, image2_.cols, image2_.rows);

  uchar4 *convertedImage1 = new uchar4[image1_.cols * image1_.rows];
  for (int i = 0; i < image1_.rows; ++i) {
    for (int j = 0; j < image1_.cols; ++j) {
      cv::Vec3b pixel = image1_.at<cv::Vec3b>(i, j);
      convertedImage1[i * image1_.cols + j] =
          make_uchar4(pixel[0], pixel[1], pixel[2], 0);
    }
  }
  cudaMemcpyToArray(cuArray1, 0, 0, convertedImage1,
                    image1_.cols * image1_.rows * sizeof(uchar4),
                    cudaMemcpyHostToDevice);
  delete convertedImage1;

  uchar4 *convertedImage2 = new uchar4[image2_.cols * image2_.rows];
  for (int i = 0; i < image2_.rows; ++i) {
    for (int j = 0; j < image2_.cols; ++j) {
      cv::Vec3b pixel = image2_.at<cv::Vec3b>(i, j);
      convertedImage2[i * image2_.cols + j] =
          make_uchar4(pixel[0], pixel[1], pixel[2], 0);
    }
  }
  cudaMemcpyToArray(cuArray2, 0, 0, convertedImage2,
                    image2_.cols * image2_.rows * sizeof(uchar4),
                    cudaMemcpyHostToDevice);
  delete convertedImage2;

  setupTexture(cuArray1, cuArray2);
}

CudaHarrisKeypointMatcher::~CudaHarrisKeypointMatcher() {
  cudaUnbindTexture(texImage1);
  cudaUnbindTexture(texImage2);
}

__global__ void matchKeypointsKernel(const float *kpsL_x, const float *kpsL_y,
                                     const float *kpsR_x, const float *kpsR_y,
                                     int numImage1Rows, int numImage1Cols,
                                     int numImage2Rows, int numImage2Cols,
                                     int numKpsL, int numKpsR,
                                     int *bestMatchIndices,
                                     uint64_t *bestMatchSSDs) {
  // Assuming each block processes one keypoint from keypointsL
  int keypointIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (keypointIdx >= numKpsL)
    return;

  int patchSize = cuPatchSize;

  float pos1_x = kpsL_x[keypointIdx];
  float pos1_y = kpsL_y[keypointIdx];
  const int border = patchSize / 2;

  // SSD calculation for the current keypoint against all keypoints in
  // keypointsR
  int bestMatchIndex = -1;
  uint64_t bestMatchSSD = 0xffffffffffffffff;
  for (int j = 0; j < numKpsR; ++j) {
    float pos2_x = kpsR_x[j];
    float pos2_y = kpsR_y[j];

    // Compute SSD
    uint64_t ssd = 0;
    for (int dy = -border; dy <= border; ++dy) {
      for (int dx = -border; dx <= border; ++dx) {
        uchar4 p1 = tex2D(texImage1, pos1_x + dx, pos1_y + dy);
        uchar4 p2 = tex2D(texImage2, pos2_x + dx, pos2_y + dy);

        ssd += (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) +
               (p1.z - p2.z) * (p1.z - p2.z);
      }
    }

    if (ssd < bestMatchSSD) {
      bestMatchSSD = ssd;
      bestMatchIndex = j;
    }
  }

  // Store the best match for this keypoint
  if (bestMatchSSD < cuMaxSSDThreash) {
    bestMatchIndices[keypointIdx] = bestMatchIndex;
    bestMatchSSDs[keypointIdx] = bestMatchSSD;
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

  int threadsPerBlock = 256;
  int numBlocks = (keypointsL.size() + threadsPerBlock - 1) / threadsPerBlock;

  matchKeypointsKernel<<<numBlocks, threadsPerBlock>>>(
      d_kpsL_X, d_kpsL_Y, d_kpsR_X, d_kpsR_Y, image1_.rows, image1_.cols,
      image2_.rows, image2_.cols, keypointsL.size(), keypointsR.size(),
      d_bestMatchIndices, d_bestMatchSSDs);
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
