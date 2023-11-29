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
  SeqHarrisCornerDetector(HarrisCornerOptions options);

  std::vector<cv::KeyPoint> detect(const cv::Mat &image) override;
};

class OcvHarrisCornerDetector : public FeatureDetector {
private:
  const HarrisCornerOptions options_;

public:
  OcvHarrisCornerDetector(HarrisCornerOptions options);

  std::vector<cv::KeyPoint> detect(const cv::Mat &image) override;
};

class MPIHarrisCornerDetector : public FeatureDetector {
private:
  const HarrisCornerOptions options_;
  const int nproc_;
  const int pid_;

public:
  MPIHarrisCornerDetector(HarrisCornerOptions options, int pid, int nproc);

  std::vector<cv::KeyPoint> detect(const cv::Mat &image) override;
};

#endif
