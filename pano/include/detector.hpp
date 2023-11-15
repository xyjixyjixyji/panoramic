#ifndef PANO_DET_HPP
#define PANO_DET_HPP

#include <opencv2/core/mat.hpp>
#include <vector>

// feature detector interface
class FeatureDetector {
public:
  virtual std::vector<cv::KeyPoint> detect(const cv::Mat &image) = 0;
  virtual ~FeatureDetector() {}
};

class SeqHarrisCornerDetector : public FeatureDetector {
public:
  SeqHarrisCornerDetector() {}

  std::vector<cv::KeyPoint> detect(const cv::Mat &image) override;

  static std::unique_ptr<FeatureDetector> createDetector() {
    return std::make_unique<SeqHarrisCornerDetector>();
  }
};

#endif
