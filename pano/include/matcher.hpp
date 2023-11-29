#ifndef PANO_MATCHER_HPP
#define PANO_MATCHER_HPP

#include "options.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

// KeyPoint matcher interface
class KeyPointMatcher {
public:
  virtual std::vector<cv::DMatch>
  matchKeyPoints(std::vector<cv::KeyPoint> keypointsL,
                 std::vector<cv::KeyPoint> keypointsR) = 0;
  virtual ~KeyPointMatcher() {}
};

class OcvHarrisKeypointMatcher : public KeyPointMatcher {
private:
  const cv::Mat &imageL_;
  const cv::Mat &imageR_;

public:
  // Initialize the matcher with two images and their keypoints
  OcvHarrisKeypointMatcher(cv::Mat &imageL, cv::Mat &imageR);

  // Match keypoints detected by Harris corner detector
  std::vector<cv::DMatch>
  matchKeyPoints(std::vector<cv::KeyPoint> keypointsL,
                 std::vector<cv::KeyPoint> keypointsR) override;
};

class SeqHarrisKeyPointMatcher : public KeyPointMatcher {
private:
  const cv::Mat &image1_;
  const cv::Mat &image2_;
  const HarrisCornerOptions options_;

public:
  // Initialize the matcher with two images and their keypoints
  SeqHarrisKeyPointMatcher(cv::Mat &image1, cv::Mat &image2,
                           HarrisCornerOptions options);

  // Match keypoints detected by Harris corner detector
  std::vector<cv::DMatch>
  matchKeyPoints(std::vector<cv::KeyPoint> keypointsL,
                 std::vector<cv::KeyPoint> keypointsR) override;
};

#endif
