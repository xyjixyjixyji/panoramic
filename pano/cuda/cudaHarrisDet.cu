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

__global__ void convolveKernel(const double* input, double* output, 
  const double* kernel, const int kernelSize, const int inputRow, const int inputCol) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;

    int k = kernelSize / 2;
    if (y < k || y >= inputRow - k) return;
    if (x < k || x >= inputCol - k) return;

    double sum = 0.0;
    for (int i = -k; i <= k; i++) {
      for (int j = -k; j <= k; j++) {
        sum += input[(y + i) * inputCol + x + j] * kernel[(k + i) * kernelSize + k + j];
      }
    }
    output[y * inputCol + x] = sum;
}

__global__ void harrisRespKernel(const double* XX, const double* YY, const double* XY, 
  double* harrisResp, const int size, const int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    double det = XX[idx] * YY[idx] - XY[idx] * XY[idx];
    double trace = XX[idx] + YY[idx];
    harrisResp[idx] = det - k * trace * trace;
}

__global__ void findKeypointKernel(const double* harrisResp, bool* isKeypoint, 
  const int thresh, const int halfLen, const int inputRow, const int inputCol) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < halfLen || y >= inputRow - halfLen) return;
    if (x < halfLen || x >= inputCol - halfLen) return;

    double resp = harrisResp[y * inputCol + x];
    if (resp <= thresh)
      return;

    double max_resp, cur_resp;
    (*((uint64_t*)&max_resp))= ~(1LL<<52);

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
      flatMat[i*col + j] = mat[i][j];
    }
  }
  return flatMat;
}

double* elemWiseMul(std::vector<double> &first, std::vector<double> &second, int size) {
  double* res = new double[size];
  for (int i = 0; i < size; ++i) {
    res[i] = first[i] * second[i];
  }
  return res;
}

/**
 * @brief Detect keypoints in the input image by Harris corner method
 *
 * @param image input image is rather a grayscale image or a BGR image
 * @return std::vector<cv::KeyPoint> the keypoints detected
 */
std::vector<cv::KeyPoint>
CudaHarrisCornerDetector::detect(const cv::Mat &image) {

  /* =============================== PART 1: CONVERT IMAGE =======================*/
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

  /* =============================== PART 2: CONVOLVE =======================*/
  dim3 blockSize(16, 16);
  dim3 gridSize((imgRow + blockSize.x - 1) / blockSize.x, 
                (imgCol + blockSize.y - 1) / blockSize.y);

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
          flatSobelX.size() * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_flatSobelY, flatSobelY.size() * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_flatSobelY, flatSobelY.data(), 
          flatSobelY.size() * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_flatGaussian, flatGaussian.size() * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_flatGaussian, flatGaussian.data(), 
          flatGaussian.size() * sizeof(double), cudaMemcpyHostToDevice));

  /* Convolve */
  double *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, imgSize * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_output, imgSize * sizeof(double)));

  // Get gradX
  CUDA_CHECK(cudaMemcpy(d_input, img, imgSize * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_output, 0.0, imgSize * sizeof(double)));
  convolveKernel<<<gridSize, blockSize>>>(d_input, d_output, d_flatSobelX, 
    sobelXKernel.size(), imgRow, imgCol);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> gradX(imgSize);
  CUDA_CHECK(cudaMemcpy(gradX.data(), d_output, imgSize * sizeof(double), cudaMemcpyDeviceToHost));

  // Get gradY
  CUDA_CHECK(cudaMemset(d_output, 0.0, imgSize * sizeof(double)));
  convolveKernel<<<gridSize, blockSize>>>(d_input, d_output, d_flatSobelY, 
    sobelYKernel.size(), imgRow, imgCol);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> gradY(imgSize);
  CUDA_CHECK(cudaMemcpy(gradY.data(), d_output, imgSize * sizeof(double), cudaMemcpyDeviceToHost));

  // Calculate XX YY XY
  double* gradXX = elemWiseMul(gradX, gradX, imgSize);
  double* gradYY = elemWiseMul(gradY, gradY, imgSize);
  double* gradXY = elemWiseMul(gradX, gradY, imgSize);

  // Get gradXX
  CUDA_CHECK(cudaMemcpy(d_input, gradXX, imgSize * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_output, 0.0, imgSize * sizeof(double)));
  convolveKernel<<<gridSize, blockSize>>>(d_input, d_output, d_flatGaussian, 
    gaussianKernel.size(), imgRow, imgCol);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(gradXX, d_output, imgSize * sizeof(double), cudaMemcpyDeviceToHost));

  // Get gradYY
  CUDA_CHECK(cudaMemcpy(d_input, gradYY, imgSize * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_output, 0.0, imgSize * sizeof(double)));
  convolveKernel<<<gridSize, blockSize>>>(d_input, d_output, d_flatGaussian, 
    gaussianKernel.size(), imgRow, imgCol);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(gradYY, d_output, imgSize * sizeof(double), cudaMemcpyDeviceToHost));

  // Get gradXY
  CUDA_CHECK(cudaMemcpy(d_input, gradXY, imgSize * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_output, 0.0, imgSize * sizeof(double)));
  convolveKernel<<<gridSize, blockSize>>>(d_input, d_output, d_flatGaussian, 
    gaussianKernel.size(), imgRow, imgCol);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(gradXY, d_output, imgSize * sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_flatSobelX));
  CUDA_CHECK(cudaFree(d_flatSobelY));
  CUDA_CHECK(cudaFree(d_flatGaussian));

  /*=============================== PART 3: BUILD HARRISRESP ===============================*/
  int threadsPerBlock = 256;
  int numBlocks = (imgSize + threadsPerBlock - 1) / threadsPerBlock;

  double *d_XX, *d_YY, *d_XY;
  CUDA_CHECK(cudaMalloc(&d_XX, imgSize * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_YY, imgSize * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_XY, imgSize * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_XX, gradXX, imgSize * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_YY, gradYY, imgSize * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_XY, gradXY, imgSize * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_output, 0.0, imgSize * sizeof(double)));

  double* harrisResp = new double[imgSize];
  harrisRespKernel<<<numBlocks, threadsPerBlock>>>(d_XX, d_YY, d_XY, d_output, imgSize, options_.k_);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(harrisResp, d_output, imgSize * sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_XX));
  CUDA_CHECK(cudaFree(d_YY));
  CUDA_CHECK(cudaFree(d_XY));
  CUDA_CHECK(cudaFree(d_output));

  /*=============================== PART 4: FIND KEYPOINTS ===============================*/
  // Non-maximum suppression
  bool *d_isKeypoint; // set to all false initially
  CUDA_CHECK(cudaMalloc(&d_isKeypoint, imgSize * sizeof(bool)));
  CUDA_CHECK(cudaMemset(d_isKeypoint, false, imgSize * sizeof(bool)));
  CUDA_CHECK(cudaMemcpy(d_input, harrisResp, imgSize * sizeof(double), cudaMemcpyHostToDevice));

  bool* isKeypoint = new bool[imgSize];
  findKeypointKernel<<<gridSize, blockSize>>>(d_input, d_isKeypoint,
    options_.nmsThresh_, options_.nmsNeighborhood_ / 2, imgRow, imgCol);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(isKeypoint, d_isKeypoint, imgSize * sizeof(bool), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_input));
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
