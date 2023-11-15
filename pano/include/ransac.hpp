#ifndef PANO_RANSAC_HPP
#define PANO_RANSAC_HPP

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

// RANSAC uses a series of random samples to generate a homography matrix.
class RansacHomographyCalculator {
public:
  virtual cv::Mat computeHomography() = 0;
  virtual ~RansacHomographyCalculator() {}
};

class SeqRansacHomographyCalculator : public RansacHomographyCalculator {
private:
  std::vector<cv::KeyPoint> keypoints1_;
  std::vector<cv::KeyPoint> keypoints2_;
  std::vector<cv::DMatch> matches_; // kp1 -> kp2{

public:
  SeqRansacHomographyCalculator(std::vector<cv::KeyPoint> &keypoints1,
                                std::vector<cv::KeyPoint> &keypoints2,
                                std::vector<cv::DMatch> &matches)
      : keypoints1_(keypoints1), keypoints2_(keypoints2), matches_(matches) {}

  // compute the homography matrix
  cv::Mat computeHomography() override;

  std::unique_ptr<RansacHomographyCalculator>
  createHomographyCalculator(std::vector<cv::KeyPoint> &keypoints1,
                             std::vector<cv::KeyPoint> &keypoints2,
                             std::vector<cv::DMatch> &matches) {
    return std::make_unique<SeqRansacHomographyCalculator>(keypoints1,
                                                           keypoints2, matches);
  }
};

#endif
