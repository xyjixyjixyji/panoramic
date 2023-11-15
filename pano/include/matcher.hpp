#ifndef PANO_MATCHER_HPP
#define PANO_MATCHER_HPP

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

// KeyPoint matcher interface
class KeyPointMatcher {
public:
  virtual std::vector<cv::DMatch> matchKeyPoints() = 0;
  virtual ~KeyPointMatcher() {}
};

class SeqHarrisKeyPointMatcher : public KeyPointMatcher {
private:
  const cv::Mat &image1_;
  const cv::Mat &image2_;
  const std::vector<cv::KeyPoint> &keypoints1_;
  const std::vector<cv::KeyPoint> &keypoints2_;

public:
  // Initialize the matcher with two images and their keypoints
  SeqHarrisKeyPointMatcher(cv::Mat &image1, cv::Mat &image2,
                           std::vector<cv::KeyPoint> &keypoints1,
                           std::vector<cv::KeyPoint> &keypoints2)
      : image1_(image1), image2_(image2), keypoints1_(keypoints1),
        keypoints2_(keypoints2) {}

  // Match keypoints detected by Harris corner detector
  std::vector<cv::DMatch> matchKeyPoints() override;

  // Factory method
  static std::unique_ptr<KeyPointMatcher>
  createMatcher(cv::Mat &image1, cv::Mat &image2,
                std::vector<cv::KeyPoint> &keypoints1,
                std::vector<cv::KeyPoint> &keypoints2) {
    return std::make_unique<SeqHarrisKeyPointMatcher>(image1, image2,
                                                      keypoints1, keypoints2);
  }
};

#endif
