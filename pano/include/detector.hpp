#ifndef PANO_DET_HPP
#define PANO_DET_HPP

#include "options.hpp"
#include <opencv2/core/mat.hpp>
#include <vector>

// feature detector interface
class FeatureDetector {
public:
  virtual std::vector<cv::KeyPoint> detect(const cv::Mat &image) = 0;
  virtual ~FeatureDetector() {}
};

class SeqHarrisCornerDetector : public FeatureDetector {
private:
  const HarrisCornerOptions options_;

public:
  SeqHarrisCornerDetector(HarrisCornerOptions options) : options_(options) {}

  std::vector<cv::KeyPoint> detect(const cv::Mat &image) override;

  static std::unique_ptr<FeatureDetector>
  createDetector(HarrisCornerOptions options) {
    return std::make_unique<SeqHarrisCornerDetector>(options);
  }
};

class OcvSiftDetector : public FeatureDetector {
public:
  OcvSiftDetector() {}

  std::vector<cv::KeyPoint> detect(const cv::Mat &image) override;

  static std::unique_ptr<FeatureDetector> createDetector() {
    return std::make_unique<OcvSiftDetector>();
  }
};

#endif
