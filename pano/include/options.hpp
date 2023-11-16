#ifndef PANO_OPTIONS_HPP
#define PANO_OPTIONS_HPP

#include <argparse/argparse.hpp>
#include <memory>
#include <optional>
#include <stdexcept>

const std::string HarrisDetector = "harris";

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
        .default_value(300000.)
        .action([](const std::string &value) { return std::stod(value); });

    args.add_argument("--harris-nms-neigh")
        .help("The neighborhood size for non-maximum suppression")
        .default_value(9)
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
  // # of iteration we are sampling
  int numIterations_;
  // # of samples we are using for each RANSAC iteration
  int numSamples_;
  // the distance that we are tolerating for a point to be considered an inlier
  double distanceThreshold_;

  RansacOptions() {}

  RansacOptions(argparse::ArgumentParser &args) {
    numIterations_ = args.get<int>("--ransac-num-iter");
    numSamples_ = args.get<int>("--ransac-num-samples");
    distanceThreshold_ = args.get<double>("--ransac-dist-thresh");
  }

  static void addRansacArguments(argparse::ArgumentParser &args) {
    args.add_argument("--ransac-num-iter")
        .help("The number of iterations for RANSAC")
        .default_value(500)
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
  std::optional<HarrisCornerOptions> harrisOptions_;
};

struct PanoramicOptions {
  DetectorOptions detOptions_;
  RansacOptions ransacOptions_;

  PanoramicOptions(argparse::ArgumentParser &args) {
    auto detectorType = args.get<std::string>("--detector");
    if (detectorType == HarrisDetector) {
      detOptions_.harrisOptions_ =
          std::make_optional(HarrisCornerOptions(args));
    }

    ransacOptions_ = RansacOptions(args);
  }

  static PanoramicOptions getRuntimeOptions(int argc, char **argv) {
    argparse::ArgumentParser args("Panoramic Image Stitcher");

    args.add_argument("--detector")
        .help("The type of feature detector to use: harris | ...")
        .default_value(HarrisDetector);

    // we provide all argument w/ default values so no exception will be thrown
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
