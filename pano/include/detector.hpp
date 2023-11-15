#include <opencv2/core/mat.hpp>
#include <vector>

// feature detector interface
class FeatureDetector {
public:
  virtual std::vector<cv::KeyPoint> detect(const cv::Mat &image) = 0;
  virtual ~FeatureDetector() {}
};

// forward declaration for different feature detectors
std::unique_ptr<FeatureDetector> createSeqHarrisCornerDetector();
