#ifndef PANO_RANSAC_HPP
#define PANO_RANSAC_HPP

#include "options.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

// RANSAC uses a series of random samples to generate a homography matrix.
class RansacHomographyCalculator {
public:
  virtual cv::Mat computeHomography(std::vector<cv::KeyPoint> &keypoints1,
                                    std::vector<cv::KeyPoint> &keypoints2,
                                    std::vector<cv::DMatch> &matches) = 0;
  virtual ~RansacHomographyCalculator() {}
};

class SeqRansacHomographyCalculator : public RansacHomographyCalculator {
private:
  std::vector<cv::KeyPoint> keypoints1_;
  std::vector<cv::KeyPoint> keypoints2_;
  std::vector<cv::DMatch> matches_; // kp1 -> kp2
  RansacOptions options_;

public:
  SeqRansacHomographyCalculator(RansacOptions options) : options_(options) {}

  // compute the homography matrix
  cv::Mat computeHomography(std::vector<cv::KeyPoint> &keypoints1,
                            std::vector<cv::KeyPoint> &keypoints2,
                            std::vector<cv::DMatch> &matches) override;

  static std::unique_ptr<RansacHomographyCalculator>
  createHomographyCalculator(RansacOptions options) {
    return std::make_unique<SeqRansacHomographyCalculator>(options);
  }
};

#endif
