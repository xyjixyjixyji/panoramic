#ifndef PANO_OPTIONS_HPP
#define PANO_OPTIONS_HPP

#include "warp.hpp"

#include <argparse/argparse.hpp>
#include <optional>
#include <stdexcept>

// --detector
const std::string HarrisDetector = "harris";
const std::string OpenCVSift = "OpenCVSift";

// --ransac
const std::string SeqRansac = "seq";
const std::string OcvRansac = "ocv";

struct HarrisCornerOptions {
  // Smaller k leads to more sensitive detection
  // Empirically, k is in [0.04, 0.06]
  double k_;
  // threshold for non-maximum suppression, a higher threshold will lead to less
  // keypoints
  double nmsThresh_;
  // WARN: has to be odd
  // the neighbor hood is [x-NMSNeighborhood, x+NMSNeighborhood]
  double nmsNeighborhood_;
  // WARN: has to be odd
  // the neighbor hood of the keypoint's patch
  int patchSize_;
  // the maximum Sum Squared Difference between two patches we are okay with
  double maxSSDThresh_;

  HarrisCornerOptions() {}

  HarrisCornerOptions(argparse::ArgumentParser &args) {
    k_ = args.get<double>("--harris-k");
    nmsThresh_ = args.get<double>("--harris-nms-thresh");
    nmsNeighborhood_ = args.get<int>("--harris-nms-neigh");
    patchSize_ = args.get<int>("--harris-patch-size");
    maxSSDThresh_ = args.get<double>("--harris-max-ssd");
  }

  static void addHarrisArguments(argparse::ArgumentParser &args) {
    args.add_argument("--harris-k")
        .help("The k parameter for Harris Corner Detector")
        .default_value(0.04)
        .action([](const std::string &value) { return std::stod(value); });

    args.add_argument("--harris-nms-thresh")
        .help("The threshold for non-maximum suppression")
        .default_value(50000.)
        .action([](const std::string &value) { return std::stod(value); });

    args.add_argument("--harris-nms-neigh")
        .help("The neighborhood size for non-maximum suppression")
        .default_value(3)
        .action([](const std::string &value) { return std::stoi(value); });

    args.add_argument("--harris-patch-size")
        .help("The patch size for Harris Corner Detector")
        .default_value(7)
        .action([](const std::string &value) { return std::stoi(value); });

    args.add_argument("--harris-max-ssd")
        .help("The maximum SSD between two patches we are okay with")
        .default_value(2500.)
        .action([](const std::string &value) { return std::stod(value); });
  }
};

struct RansacOptions {
  std::string ransacType_;
  // # of iteration we are sampling
  int numIterations_;
  // # of samples we are using for each RANSAC iteration
  int numSamples_;
  // the distance that we are tolerating for a point to be considered an inlier
  double distanceThreshold_;

  RansacOptions() {}

  RansacOptions(argparse::ArgumentParser &args) {
    ransacType_ = args.get<std::string>("--ransac");
    numIterations_ = args.get<int>("--ransac-num-iter");
    numSamples_ = args.get<int>("--ransac-num-samples");
    distanceThreshold_ = args.get<double>("--ransac-dist-thresh");
  }

  static void addRansacArguments(argparse::ArgumentParser &args) {
    args.add_argument("--ransac-num-iter")
        .help("The number of iterations for RANSAC")
        .default_value(1000)
        .action([](const std::string &value) { return std::stoi(value); });

    args.add_argument("--ransac-num-samples")
        .help("The number of samples for each RANSAC iteration")
        .default_value(4)
        .action([](const std::string &value) { return std::stoi(value); });

    args.add_argument("--ransac-dist-thresh")
        .help("The distance threshold for a point to be considered an inlier")
        .default_value(5.)
        .action([](const std::string &value) { return std::stod(value); });
  }
};

struct DetectorOptions {
  std::string detectorType_;
  std::optional<HarrisCornerOptions> harrisOptions_;
};

struct PanoramicOptions {
  DetectorOptions detOptions_;
  RansacOptions ransacOptions_;
  std::vector<std::string> imgPaths_;
  warpFunction_t warpFunction_;

  PanoramicOptions(argparse::ArgumentParser &args) {
    auto detectorType = args.get<std::string>("--detector");
    imgPaths_ = args.get<std::vector<std::string>>("--img");

    detOptions_.detectorType_ = detectorType;
    if (detectorType == HarrisDetector) {
      detOptions_.harrisOptions_ =
          std::make_optional(HarrisCornerOptions(args));
    } else if (detectorType == OpenCVSift) {
      // pass
    }

    ransacOptions_ = RansacOptions(args);

    auto warpType = args.get<std::string>("--warp");
    if (warpType == "sequential") {
      warpFunction_ = warpSequential;
    } else {
      throw std::runtime_error("Unsupported warp function");
    }
  }

  static PanoramicOptions getRuntimeOptions(int argc, char **argv) {
    argparse::ArgumentParser args("Panoramic Image Stitcher");

    args.add_argument("--img")
        .help("The images you want to stitch, from **left to right**")
        .append()
        .required();

    args.add_argument("--detector")
        .help("The type of feature detector to use: harris | OpenCVSift | ...")
        .default_value(HarrisDetector);

    args.add_argument("--ransac")
        .help("The type of RANSAC to use: seq | ocv")
        .default_value(SeqRansac);

    args.add_argument("--warp")
        .help("The type of warp function to use: sequential | ...")
        .default_value("sequential");

    // we provide all argument w/ default values so no exception will be
    // thrown
    HarrisCornerOptions::addHarrisArguments(args);
    RansacOptions::addRansacArguments(args);

    try {
      args.parse_args(argc, argv); // Example: ./main --color orange
    } catch (const std::exception &err) {
      std::cerr << err.what() << std::endl;
      std::cerr << args;
      std::exit(1);
    }

    return PanoramicOptions(args);
  }
};

#endif
