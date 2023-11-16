#ifndef PANO_STITCHER_HPP
#define PANO_STITCHER_HPP

#include "matcher.hpp"
#include "options.hpp"
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
  Stitcher(cv::Mat image1, cv::Mat image2, PanoramicOptions options) {
    imageL_ = image1;
    imageR_ = image2;
    auto detOptions = options.detOptions_;
    auto ransacOptions = options.ransacOptions_;

    if (detOptions.harrisOptions_ != std::nullopt) {
      detector_ = SeqHarrisCornerDetector::createDetector(
          detOptions.harrisOptions_.value());
      matcher_ = SeqHarrisKeyPointMatcher::createMatcher(
          imageL_, imageR_, detOptions.harrisOptions_.value());
    }

    homographyCalculator_ =
        SeqRansacHomographyCalculator::createHomographyCalculator(
            ransacOptions);
  }

  /**
   * @brief Stitch two images together to form a panorama
   *
   * @param image1 the first image
   * @param image2 the second image
   * @return the stitched panorama
   */
  cv::Mat stitch();

  static std::unique_ptr<Stitcher>
  createStitcher(cv::Mat imageL, cv::Mat imageR, PanoramicOptions options) {
    return std::make_unique<Stitcher>(imageL, imageR, options);
  }

private:
  // feature detector
  std::unique_ptr<FeatureDetector> detector_;
  // matcher
  std::unique_ptr<KeyPointMatcher> matcher_;
  // homography calculator
  std::unique_ptr<RansacHomographyCalculator> homographyCalculator_;

  // we have two images to be stitched
  cv::Mat imageL_;
  cv::Mat imageR_;
};

#endif
