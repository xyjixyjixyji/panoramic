#include <pano.hpp>

#include <mpi.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cassert>
#include <iostream>

int main(int argc, char **argv) {
  PanoramicOptions options = PanoramicOptions::getRuntimeOptions(argc, argv);

  /* In benchmark mode. */
  if (options.benchmark_ != "") {
    benchmark(options.benchmark_);
    return 0;
  }

  std::vector<std::string> imgPaths = options.imgPaths_;

  std::vector<cv::Mat> toWarped;
  for (auto &imgPath : imgPaths) {
    auto img = cv::imread(imgPath, cv::IMREAD_COLOR);
    toWarped.push_back(img);
  }

  cv::Mat warped = timedStitchAllSequential(toWarped, options);

  if (options.use_mpi_) {
    if (options.pid_ == 0) {
      cv::imwrite("Warped.png", warped);
    }
    MPI_Finalize();
  } else {
    cv::imwrite("Warped.png", warped);
  }

  return 0;
}
