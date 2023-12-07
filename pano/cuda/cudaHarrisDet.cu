#include <common.hpp>
#include <detector.hpp>
#include <omp.h>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
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

CudaHarrisCornerDetector::CudaHarrisCornerDetector(HarrisCornerOptions options)
    : options_(options) {}

__global__ void convolveKernel(const double *input, double *output,
                               const double *kernel, const int kernelSize,
                               const int inputRow, const int inputCol) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;

  int k = kernelSize / 2;
  if (y * inputCol + x >= inputRow * inputCol)
    return;
  if ((y < k || y >= inputRow - k) || (x < k || x >= inputCol - k)) {
    output[y * inputCol + x] = 0.0;
    return;
  }

  double sum = 0.0;
  for (int i = -k; i <= k; i++) {
    for (int j = -k; j <= k; j++) {
      sum += input[(y + i) * inputCol + x + j] *
             kernel[(k + i) * kernelSize + k + j];
    }
  }
  output[y * inputCol + x] = sum;
}

__global__ void elemWiseMulKernel(const double *x, const double *y, double *xx,
                                  double *yy, double *xy, const int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;
  xx[i] = x[i] * x[i];
  yy[i] = y[i] * y[i];
  xy[i] = x[i] * y[i];
}

__global__ void harrisRespKernel(const double *XX, const double *YY,
                                 const double *XY, double *harrisResp,
                                 const int size, const int k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  double det = XX[idx] * YY[idx] - XY[idx] * XY[idx];
  double trace = XX[idx] + YY[idx];
  harrisResp[idx] = det - k * trace * trace;
}

__global__ void findKeypointKernel(const double *harrisResp, bool *isKeypoint,
                                   const int thresh, const int halfLen,
                                   const int inputRow, const int inputCol) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;

  if (y < halfLen || y >= inputRow - halfLen)
    return;
  if (x < halfLen || x >= inputCol - halfLen)
    return;

  double resp = harrisResp[y * inputCol + x];
  if (resp <= thresh)
    return;

  double max_resp, cur_resp;
  (*((uint64_t *)&max_resp)) = ~(1LL << 52);

  for (int i = -halfLen; i <= halfLen; i++) {
    for (int j = -halfLen; j <= halfLen; j++) {
      if (i == 0 && j == 0)
        continue;
      cur_resp = harrisResp[(y + i) * inputCol + x + j];
      if (cur_resp > max_resp)
        max_resp = cur_resp;
    }
  }

  if (resp > max_resp) {
    isKeypoint[y * inputCol + x] = true;
  }
}

std::vector<double> flattenMatrix(std::vector<std::vector<double>> &mat) {
  int row = mat.size();
  int col = mat[0].size();
  std::vector<double> flatMat(row * col);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      flatMat[i * col + j] = mat[i][j];
    }
  }
  return flatMat;
}

/**
 * @brief Detect keypoints in the input image by Harris corner method
 *
 * @param image input image is rather a grayscale image or a BGR image
 * @return std::vector<cv::KeyPoint> the keypoints detected
 */
std::vector<cv::KeyPoint>
CudaHarrisCornerDetector::detect(const cv::Mat &image) {

  /* =============================== PART 1: CONVERT IMAGE
   * =======================*/
  // Ensure the image is grayscale
  cv::Mat gray;
  if (image.channels() == 3) {
    cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = image;
  }
  gray.convertTo(gray, CV_64F); // Ensure the image is in double format

  /* Convert image to a flat double array*/
  auto cudaChennelDesc = cudaCreateChannelDesc<double>();
  int imgCol = gray.cols;
  int imgRow = gray.rows;
  int imgSize = imgCol * imgRow;
  double *img = new double[imgCol * imgRow];
  for (int i = 0; i < imgRow; ++i) {
    for (int j = 0; j < imgCol; ++j) {
      img[i * imgCol + j] = gray.at<double>(i, j);
    }
  }

  /* For kernels on flattened things. */
  int threadsPerBlock = 256;
  int numBlocks = (imgSize + threadsPerBlock - 1) / threadsPerBlock;

  /* For kernels on 2D things. */
  dim3 blockSize(16, 16);
  dim3 gridSize((imgRow + blockSize.x - 1) / blockSize.x,
                (imgCol + blockSize.y - 1) / blockSize.y);

  /* =============================== PART 2: CONVOLVE =======================*/
  /* Get matrix kernel and flatten. */
  std::vector<std::vector<double>> sobelXKernel = getSobelXKernel();
  std::vector<std::vector<double>> sobelYKernel = getSobelYKernel();
  std::vector<std::vector<double>> gaussianKernel = getGaussianKernel(5, 1.0);
  std::vector<double> flatSobelX = flattenMatrix(sobelXKernel);
  std::vector<double> flatSobelY = flattenMatrix(sobelYKernel);
  std::vector<double> flatGaussian = flattenMatrix(gaussianKernel);

  /* Copy matrices to device. */
  double *d_flatSobelX, *d_flatSobelY, *d_flatGaussian;
  CUDA_CHECK(cudaMalloc(&d_flatSobelX, flatSobelX.size() * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_flatSobelX, flatSobelX.data(),
                        flatSobelX.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_flatSobelY, flatSobelY.size() * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_flatSobelY, flatSobelY.data(),
                        flatSobelY.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_flatGaussian, flatGaussian.size() * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_flatGaussian, flatGaussian.data(),
                        flatGaussian.size() * sizeof(double),
                        cudaMemcpyHostToDevice));

  /* Convolve */
  double *d_img, *d_gradX, *d_gradY;
  double *d_gradXXMid, *d_gradYYMid, *d_gradXYMid;
  double *d_gradXX, *d_gradYY, *d_gradXY;
  CUDA_CHECK(cudaMalloc(&d_img, imgSize * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_gradX, imgSize * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_gradY, imgSize * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_gradXXMid, imgSize * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_gradYYMid, imgSize * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_gradXYMid, imgSize * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_gradXX, imgSize * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_gradYY, imgSize * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_gradXY, imgSize * sizeof(double)));
  CUDA_CHECK(
      cudaMemcpy(d_img, img, imgSize * sizeof(double), cudaMemcpyHostToDevice));

  // Compute gradX
  convolveKernel<<<gridSize, blockSize>>>(d_img, d_gradX, d_flatSobelX,
                                          sobelXKernel.size(), imgRow, imgCol);
  // CUDA_CHECK(cudaDeviceSynchronize()); // TODO check if could be deleted

  // Compute gradY
  convolveKernel<<<gridSize, blockSize>>>(d_img, d_gradY, d_flatSobelY,
                                          sobelYKernel.size(), imgRow, imgCol);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Calculate element wise multiplication XX YY XY
  elemWiseMulKernel<<<numBlocks, threadsPerBlock>>>(
      d_gradX, d_gradY, d_gradXXMid, d_gradYYMid, d_gradXYMid, imgSize);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Get gradXX
  convolveKernel<<<gridSize, blockSize>>>(d_gradXXMid, d_gradXX, d_flatGaussian,
                                          gaussianKernel.size(), imgRow,
                                          imgCol);
  // CUDA_CHECK(cudaDeviceSynchronize());  // TODO check if could be deleted

  // Get gradYY
  convolveKernel<<<gridSize, blockSize>>>(d_gradYYMid, d_gradYY, d_flatGaussian,
                                          gaussianKernel.size(), imgRow,
                                          imgCol);
  // CUDA_CHECK(cudaDeviceSynchronize());  // TODO check if could be deleted

  // Get gradXY
  convolveKernel<<<gridSize, blockSize>>>(d_gradXYMid, d_gradXY, d_flatGaussian,
                                          gaussianKernel.size(), imgRow,
                                          imgCol);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_img));
  CUDA_CHECK(cudaFree(d_gradX));
  CUDA_CHECK(cudaFree(d_gradY));
  CUDA_CHECK(cudaFree(d_gradXXMid));
  CUDA_CHECK(cudaFree(d_gradYYMid));
  CUDA_CHECK(cudaFree(d_gradXYMid));
  CUDA_CHECK(cudaFree(d_flatSobelX));
  CUDA_CHECK(cudaFree(d_flatSobelY));
  CUDA_CHECK(cudaFree(d_flatGaussian));

  /*=============================== PART 3: BUILD HARRISRESP
   * ===============================*/
  double *d_harrisResp;
  CUDA_CHECK(cudaMalloc(&d_harrisResp, imgSize * sizeof(double)));

  harrisRespKernel<<<numBlocks, threadsPerBlock>>>(
      d_gradXX, d_gradYY, d_gradXY, d_harrisResp, imgSize, options_.k_);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_gradXX));
  CUDA_CHECK(cudaFree(d_gradYY));
  CUDA_CHECK(cudaFree(d_gradXY));

  /*=============================== PART 4: FIND KEYPOINTS
   * ===============================*/
  // Non-maximum suppression
  bool *d_isKeypoint; // set to all false initially
  CUDA_CHECK(cudaMalloc(&d_isKeypoint, imgSize * sizeof(bool)));
  CUDA_CHECK(cudaMemset(d_isKeypoint, false, imgSize * sizeof(bool)));

  bool *isKeypoint = new bool[imgSize];
  findKeypointKernel<<<gridSize, blockSize>>>(
      d_harrisResp, d_isKeypoint, options_.nmsThresh_,
      options_.nmsNeighborhood_ / 2, imgRow, imgCol);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(isKeypoint, d_isKeypoint, imgSize * sizeof(bool),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_harrisResp));
  CUDA_CHECK(cudaFree(d_isKeypoint));

  std::vector<cv::KeyPoint> keypoints;
  int x, y;
  for (int i = 0; i < imgSize; i++) {
    if (isKeypoint[i]) {
      x = i % imgCol;
      y = i / imgCol;
      keypoints.push_back(cv::KeyPoint(x, y, 1.f));
    }
  }
  return keypoints;
}
