#include <common.hpp>
#include <limits>
#include <matcher.hpp>
#include <opencv2/core/types.hpp>
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

CudaHarrisKeypointMatcher::CudaHarrisKeypointMatcher(
    cv::Mat &image1, cv::Mat &image2, HarrisCornerOptions options)
    : image1_(image1), image2_(image2), options_(options) {}

__global__ void matchKeypointsKernel(
    const float *kpsL_x, const float *kpsL_y, const float *kpsR_x,
    const float *kpsR_y, uchar3 *image1GPU, uchar3 *image2GPU,
    int numImage1Rows, int numImage1Cols, int numImage2Rows, int numImage2Cols,
    int numKpsL, int numKpsR, int *bestMatchIndices, double *bestMatchSSDs,
    const int patchSize, const int maxSSDThresh) {
  // cuda kernel will be responsible for kpsL[i], matching it with everypoint of
  // kpsR, store the best match in bestMatchIndices[i] and bestMatchSSDs[i]
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numKpsL)
    return;

  const int border = patchSize / 2;

  float pos1_x = kpsL_x[i];
  float pos1_y = kpsL_y[i];
  if (pos1_x < border || pos1_y < border || pos1_x + border >= numImage1Cols ||
      pos1_y + border >= numImage1Rows) {
    return;
  }

  int bestMatchIndex = -1;
  double bestMatchSSD = 1e300;
  for (size_t j = 0; j < numKpsR; j++) {
    float pos2_x = kpsR_x[j];
    float pos2_y = kpsR_y[j];
    if (pos2_x < border || pos2_y < border ||
        pos2_x + border >= numImage2Cols || pos2_y + border >= numImage2Rows) {
      continue;
    }

    double ssd = 0;
    for (int y = -border; y <= border; y++) {
      for (int x = -border; x <= border; x++) {
        uchar3 p1 =
            image1GPU[(int)(pos1_y + y) * numImage1Cols + (int)(pos1_x + x)];
        uchar3 p2 =
            image2GPU[(int)(pos2_y + y) * numImage2Cols + (int)(pos2_x + x)];
        double diff = 0;
        diff += (p1.x - p2.x) * (p1.x - p2.x);
        diff += (p1.y - p2.y) * (p1.y - p2.y);
        diff += (p1.z - p2.z) * (p1.z - p2.z);
        ssd += pow(diff, 2);
        if (ssd > bestMatchSSD) {
          goto end;
        }
      }
    }

  end:
    if (ssd < bestMatchSSD) {
      bestMatchSSD = ssd;
      bestMatchIndex = j;
    }
  }

  if (bestMatchSSD < maxSSDThresh) {
    bestMatchIndices[i] = bestMatchIndex;
    bestMatchSSDs[i] = bestMatchSSD;
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
  const double maxSSDThresh = options_.maxSSDThresh_;

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
  double *d_bestMatchSSDs;
  CUDA_CHECK(cudaMalloc(&d_bestMatchIndices, keypointsL.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_bestMatchSSDs, keypointsL.size() * sizeof(double)));

  // Initialize the best match arrays to -1 (or any sentinel value)
  CUDA_CHECK(
      cudaMemset(d_bestMatchIndices, -1, keypointsL.size() * sizeof(int)));
  CUDA_CHECK(
      cudaMemset(d_bestMatchSSDs, -1, keypointsL.size() * sizeof(double)));

  int threadPerBlock = 256;
  int numBlock = (keypointsL.size() + threadPerBlock - 1) / threadPerBlock;
  matchKeypointsKernel<<<numBlock, threadPerBlock>>>(
      d_kpsL_X, d_kpsL_Y, d_kpsR_X, d_kpsR_Y, d_image1, d_image2, image1_.rows,
      image1_.cols, image2_.rows, image2_.cols, keypointsL.size(),
      keypointsR.size(), d_bestMatchIndices, d_bestMatchSSDs, patchSize,
      maxSSDThresh);

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
