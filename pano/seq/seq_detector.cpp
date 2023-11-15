#include <detector.hpp>

#include <opencv2/core/types.hpp>
#include <vector>

class SeqHarrisCornerDetector : public FeatureDetector {
public:
  std::vector<cv::KeyPoint> detect(const cv::Mat &image) override;
  std::unique_ptr<FeatureDetector> createSeqHarrisCornerDetector() {
    return std::make_unique<SeqHarrisCornerDetector>();
  }
};

std::vector<cv::KeyPoint>
SeqHarrisCornerDetector::detect(const cv::Mat &image) {
  std::vector<cv::KeyPoint> keypoints;

  return keypoints;
}
