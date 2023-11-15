#ifndef PANO_STITCHER_HPP
#define PANO_STITCHER_HPP

#include "matcher.hpp"
#include "ransac.hpp"
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

#include <vector>

#include <detector.hpp>

/**
 * @brief Stitcher is the class that stitches two images together to form a
 * panorama.
 *
 * Steps:
 *  1. Detect keypoints in both images, and compute descriptors for each
 * keypoint
 *  2. Match the descriptors between the two images
 *  3. Use the matches to estimate the homography between the two images(RANSAC)
 *  4. Warp the second image to align with the first image
 *  5. (optional) Blend the two images together
 */
class Stitcher {
public:
  /**
   * @brief Construct a new Stitcher object
   *
   * @param detector
   * @param matcher
   * @param homographyCalculator
   * @param image1
   * @param image2
   */
  Stitcher(std::unique_ptr<FeatureDetector> &detector,
           std::unique_ptr<KeyPointMatcher> &matcher,
           std::unique_ptr<RansacHomographyCalculator> &homographyCalculator,
           cv::Mat image1, cv::Mat image2)
      : detector_(std::move(detector)), matcher_(std::move(matcher)),
        homographyCalculator_(std::move(homographyCalculator)), image1_(image1),
        image2_(image2) {}

  /**
   * @brief Stitch two images together to form a panorama
   *
   * @param image1 the first image
   * @param image2 the second image
   * @return the stitched panorama
   */
  cv::Mat stitch(const cv::Mat &imageL, const cv::Mat &imageR);

  static std::unique_ptr<Stitcher> createStitcher(
      std::unique_ptr<FeatureDetector> &detector,
      std::unique_ptr<KeyPointMatcher> &matcher,
      std::unique_ptr<RansacHomographyCalculator> &homographyCalculator,
      cv::Mat image1, cv::Mat image2) {
    return std::make_unique<Stitcher>(detector, matcher, homographyCalculator,
                                      image1, image2);
  }

private:
  // feature detector
  std::unique_ptr<FeatureDetector> detector_;
  // matcher
  std::unique_ptr<KeyPointMatcher> matcher_;
  // homography calculator
  std::unique_ptr<RansacHomographyCalculator> homographyCalculator_;

  // we have two images to be stitched
  cv::Mat image1_;
  cv::Mat image2_;
};

#endif
