#include <cassert>
#include <common.hpp>
#include <cstdint>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <stitcher.hpp>
#include <vector>
#include <cstdlib>
#include <stdio.h>

/* For benchmark. */
const int datapointNum = 9;
char buffer[80];

std::vector<std::vector<double>> getSobelXKernel() {
  return {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
}

std::vector<std::vector<double>> getSobelYKernel() {
  return {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
}

std::vector<std::vector<double>> getGaussianKernel(int kernelSize,
                                                   double sigma) {
  std::vector<std::vector<double>> kernel(kernelSize,
                                          std::vector<double>(kernelSize));
  double sum = 0;
  for (int i = 0; i < kernelSize; ++i) {
    int x = i - kernelSize / 2;
    for (int j = 0; j < kernelSize; ++j) {
      int y = j - kernelSize / 2;
      kernel[i][j] = exp(-(x * x + y * y) / (2 * sigma * sigma));
      sum += kernel[i][j];
    }
  }
  for (auto &row : kernel) {
    for (auto &elem : row) {
      elem /= sum;
    }
  }
  return kernel;
}

/**
 * @brief Convolve the input image with the kernel
 *
 * @param input a grayscale image
 * @param kernel
 * @return cv::Mat output
 */
cv::Mat convolveSequential(const cv::Mat &input,
                           const std::vector<std::vector<double>> kernel) {
  // kernel size has to be odd
  int kernelSize = kernel.size();
  assert(kernelSize % 2 == 1 && "Kernel size has to be odd");

  int k = kernelSize / 2;
  cv::Mat output(input.rows, input.cols, CV_64FC1);

  for (int y = k; y < input.rows - k; y++) {
    for (int x = k; x < input.cols - k; x++) {
      double sum = 0.0;
      for (int i = -k; i <= k; i++) {
        for (int j = -k; j <= k; j++) {
          sum += input.at<double>(y + i, x + j) * kernel[k + i][k + j];
        }
      }

      output.at<double>(y, x) = sum;
    }
  }

  return output;
}

inline cv::Mat __stitchTwo(cv::Mat imageL, cv::Mat imageR,
                           PanoramicOptions options) {
  auto stitcher = Stitcher::createStitcher(imageL, imageR, options);
  auto warped = stitcher->stitch();
  return warped;
}

cv::Mat __stitchAllSequential(std::vector<cv::Mat> images,
                              PanoramicOptions options) {
  if (images.size() == 1) {
    return images[0];
  }

  if (images.size() == 2) {
    return __stitchTwo(images[0], images[1], options);
  }

  std::vector<cv::Mat> toWarped;
  for (size_t i = 0; i < images.size(); i += 2) {
    if (i == images.size() - 1) {
      // we have one more images
      toWarped.push_back(images[i]);
    } else {
      toWarped.push_back(__stitchTwo(images[i], images[i + 1], options));
    }
  }

  return __stitchAllSequential(toWarped, options);
}

cv::Mat timedStitchAllSequential(std::vector<cv::Mat> images,
                                 PanoramicOptions options) {
  bool shouldPrint =
      !options.use_mpi_ || (options.use_mpi_ && options.pid_ == 0);

  Timer timer("Time to stitch all images", shouldPrint);
  return __stitchAllSequential(images, options);
}

void parseResult(std::vector<double> &store) {
  FILE *file = fopen("output.txt", "r");

  double myDouble;
  int myInt;
  for (int i = 0; i < datapointNum; i++) {
    if (fgets(buffer, sizeof(buffer), file) != NULL) {
        if (sscanf(buffer, "%*[^0123456789.-]%lf", &myDouble) == 1) {
            store[i] += myDouble;
        } else if (sscanf(buffer, "%d", &myInt) == 1) {
            store[i] += myInt;
        } else {
            i--;
        }
    } else {
      std::cout << ("missing datapoints!") << std::endl;
      break;
    }
  }

  fclose(file);
}

void printArray(std::vector<double> &data, int iter, std::string prompt) {
  std::cout << prompt;
  for (int i = 0; i < datapointNum; ++i) {
    std::cout << data[i] / iter << " ";
  }
  std::cout << std::endl;
}

void launchSequentialBenchmark(int iter, std::string& imgPathes, 
  std::string& redirect, std::vector<double> &data) {
  std::string cmd = "./build/pano_cmd --detector seq --ransac seq";

  for (int i = 0; i < iter; i++) {
    std::system((cmd + imgPathes + redirect).c_str());
    parseResult(data);
  }
}

void launchCudaBenchmark(int iter, std::string& imgPathes, 
  std::string& redirect, std::vector<double> &data) {
  std::string cmd = "./build/pano_cmd --detector cuda --ransac ocv";

  for (int i = 0; i < iter; i++) {
    std::system((cmd + imgPathes + redirect).c_str());
    parseResult(data);
  }
}

void launchOcvBenchmark(int iter, std::string& imgPathes, 
  std::string& redirect, std::vector<double> &data) {
  std::string cmd = "./build/pano_cmd --detector ocv --ransac ocv";

  for (int i = 0; i < iter; i++) {
    std::system((cmd + imgPathes + redirect).c_str());
    parseResult(data);
  }
}

void launchOmpBenchmark(int iter, std::string& imgPathes, 
  std::string& redirect, std::vector<double> &data, std::string num_threads) {
  std::string set_thread_num = "OMP_NUM_THREADS=";
  std::string cmd = " && ./build/pano_cmd --detector omp --ransac omp";

  for (int i = 0; i < iter; i++) {
    std::system((set_thread_num + num_threads + cmd + imgPathes + redirect).c_str());
    parseResult(data);
  }
}

void launchMpiBenchmark(int iter, std::string& imgPathes, 
  std::string& redirect, std::vector<double> &data, std::string num_threads) {
  std::string set_thread_num = "mpirun -n ";
  std::string cmd = " ./build/pano_cmd --detector mpi --ransac mpi";

  for (int i = 0; i < iter; i++) {
    std::system((set_thread_num + num_threads + cmd + imgPathes + redirect).c_str());
    parseResult(data);
  }
}

void benchmark(std::string machine) {
  bool ghc = machine == "ghc";
  int iter = 5;
  std::string imgPrompt = " --img ./data/";
  std::string redirect = "> output.txt 2> mpi.suppress";
  std::vector<std::string> threadCount = {"2", "3", "4", "5", "6", "7", "8"};

  if (!ghc) {
      threadCount = {"16", "32", "64", "128"};
  }

  // Available tasks
  std::vector<std::vector<std::string>> tasks = {
    {"Random Lines - 2 Img, Sparse KeyPt, High Matching", "random1.png", "random2.png"},
    {"Space - 2 Img, Sparse KeyPt, Low Matching", "space1.jpg", "space2.jpg"},
    {"View - 2 Img, Dense KeyPt, High Matching", "viewL.png", "viewR.png"},
    {"View - 4 Img, Dense KeyPt, High Matching", "v1.png", "v2.png", "v3.png", "v4.png"},
    {"Bird - 3 Img, Dense KeyPt, Low Matching", "bird1.jpg", "bird2.jpg", "bird3.jpg"},
  };

  for (std::vector<std::string> task : tasks) {
    std::cout << "Task - " << task[0] << std::endl;

    std::vector<double> seq(datapointNum, 0.0);
    std::vector<double> cuda(datapointNum, 0.0);
    std::vector<double> ocv(datapointNum, 0.0);
    std::vector<std::vector<double>> omp(threadCount.size(), std::vector<double>(datapointNum, 0.0));
    std::vector<std::vector<double>> mpi(threadCount.size(), std::vector<double>(datapointNum, 0.0));

    for (size_t i = 1; i < task.size() - 1; i++) {
      // Build image path
      std::string imgPathes = imgPrompt + task[i] + imgPrompt + task[i + 1];

      // Launch benchmarks
      if (ghc) {
        launchSequentialBenchmark(iter, imgPathes, redirect, seq);
        launchCudaBenchmark(iter, imgPathes, redirect, cuda);
        launchOcvBenchmark(iter, imgPathes, redirect, ocv);
      }

      for (size_t i = 0; i < threadCount.size(); i++) {
        launchOmpBenchmark(iter, imgPathes, redirect, omp[i], threadCount[i]);
        launchMpiBenchmark(iter, imgPathes, redirect, mpi[i], threadCount[i]);
      }
    }

    // Print results
    if (ghc) {
      printArray(seq, iter, "seq: ");
      printArray(cuda, iter, "cuda: ");
      printArray(ocv, iter, "ocv: ");
    }
    for (size_t i = 0; i < threadCount.size(); i++) {
      printArray(omp[i], iter, "omp_" + threadCount[i] + ": ");
    }
    for (size_t i = 0; i < threadCount.size(); i++) {
      printArray(mpi[i], iter, "mpi_" + threadCount[i] + ": ");
    }
    std::cout << std::endl;
  }

  std::system("rm output.txt");
  std::system("rm mpi.suppress");
}

std::vector<cv::KeyPoint>
seqHarrisCornerDetectorDetect(const cv::Mat &image,
                              HarrisCornerOptions options) {
  // options
  const double k = options.k_;
  const double thresh = options.nmsThresh_;
  const double NMSNeighborhood = options.nmsNeighborhood_;

  std::vector<cv::KeyPoint> keypoints;
  auto sobelXKernel = getSobelXKernel();
  auto sobelYKernel = getSobelYKernel();
  auto gaussianKernel = getGaussianKernel(5, 1.0);

  // Ensure the image is grayscale
  cv::Mat gray;
  if (image.channels() == 3) {
    cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = image;
  }
  gray.convertTo(gray, CV_64F); // Ensure the image is in double format

  cv::Mat gradX = convolveSequential(gray, sobelXKernel);
  cv::Mat gradY = convolveSequential(gray, sobelYKernel);
  cv::Mat gradXX = gradX.mul(gradX);
  cv::Mat gradYY = gradY.mul(gradY);
  cv::Mat gradXY = gradX.mul(gradY);

  gradXX = convolveSequential(gradXX, gaussianKernel);
  gradYY = convolveSequential(gradYY, gaussianKernel);
  gradXY = convolveSequential(gradXY, gaussianKernel);

  cv::Mat harrisResp = cv::Mat(gray.size(), CV_64F);
  for (int y = 0; y < gray.rows; y++) {
    for (int x = 0; x < gray.cols; x++) {
      double xx = gradXX.at<double>(y, x);
      double yy = gradYY.at<double>(y, x);
      double xy = gradXY.at<double>(y, x);
      double det = xx * yy - xy * xy;
      double trace = xx + yy;
      harrisResp.at<double>(y, x) = det - k * trace * trace;
    }
  }

  // Non-maximum suppression
  int halfLen = NMSNeighborhood / 2;
  for (int y = halfLen; y < gray.rows; y++) {
    for (int x = halfLen; x < gray.cols; x++) {
      double resp = harrisResp.at<double>(y, x);
      if (resp <= thresh)
        continue;

      // find the max around this point
      double max_resp = std::numeric_limits<double>::min();
      for (int i = -halfLen; i <= halfLen; i++) {
        for (int j = -halfLen; j <= halfLen; j++) {
          if (i == 0 && j == 0)
            continue;
          max_resp = std::max(max_resp, harrisResp.at<double>(y + i, x + j));
          if (max_resp > resp)
            goto bail;
        }
      }

      if (resp > max_resp) {
        keypoints.push_back(cv::KeyPoint(x, y, 1.f));
      }
    bail : {}
    }
  }

  return keypoints;
}

std::vector<cv::DMatch>
seqHarrisMatchKeyPoints(std::vector<cv::KeyPoint> keypointsL,
                        std::vector<cv::KeyPoint> keypointsR,
                        const cv::Mat &image1, const cv::Mat &image2,
                        const HarrisCornerOptions options, int offset) {
  // options
  const int patchSize = options.patchSize_;
  const double maxSSDThresh = options.maxSSDThresh_;

  std::vector<cv::DMatch> matches;
  int border = patchSize / 2;

  for (size_t i = 0; i < keypointsL.size(); i++) {
    const auto &kp1 = keypointsL[i];
    cv::Point2f pos1 = kp1.pt;

    if (pos1.x < border || pos1.y < border || pos1.x + border >= image1.cols ||
        pos1.y + border >= image1.rows) {
      continue;
    }

    int bestMatchIndex = -1;
    uint64_t bestMatchSSD = std::numeric_limits<uint64_t>::max();
    for (size_t j = 0; j < keypointsR.size(); j++) {
      const auto &kp2 = keypointsR[j];
      cv::Point2f pos2 = kp2.pt;

      if (pos2.x < border || pos2.y < border ||
          pos2.x + border >= image2.cols || pos2.y + border >= image2.rows) {
        continue;
      }

      uint64_t ssd = 0;
      for (int y = -border; y <= border; y++) {
        for (int x = -border; x <= border; x++) {
          cv::Vec3b p1 = image1.at<cv::Vec3b>(pos1.y + y, pos1.x + x);
          cv::Vec3b p2 = image2.at<cv::Vec3b>(pos2.y + y, pos2.x + x);
          uint64_t diff = 0;
          for (int c = 0; c < 3; c++) {
            diff += (p1[c] - p2[c]) * (p1[c] - p2[c]);
          }
          ssd += diff * diff;
        }
      }

      if (ssd < bestMatchSSD) {
        bestMatchSSD = ssd;
        bestMatchIndex = j;
      }
    }

    if (bestMatchSSD < maxSSDThresh) {
      matches.push_back(cv::DMatch(i + offset, bestMatchIndex, bestMatchSSD));
    }
  }

  return matches;
}