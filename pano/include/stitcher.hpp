#ifndef PANO_STITCHER_HPP
#define PANO_STITCHER_HPP

#include <cassert>
#include <common.hpp>
#include <detector.hpp>
#include <matcher.hpp>
#include <options.hpp>
#include <ransac.hpp>
#include <warp.hpp>

#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

#include <vector>

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
  Stitcher(cv::Mat image1, cv::Mat image2, PanoramicOptions options)
      : options_(options) {
    imageL_ = image1;
    imageR_ = image2;
    auto detOptions = options.detOptions_;
    auto ransacOptions = options.ransacOptions_;

    if (detOptions.detectorType_ == SeqHarrisDetector) {
      detector_ = std::make_unique<SeqHarrisCornerDetector>(
          SeqHarrisCornerDetector(detOptions.harrisOptions_.value()));
      matcher_ = std::make_unique<SeqHarrisKeyPointMatcher>(
          imageL_, imageR_, detOptions.harrisOptions_.value());
    } else if (detOptions.detectorType_ == OpenCVHarrisDetector) {
      detector_ = std::make_unique<OcvHarrisCornerDetector>(
          OcvHarrisCornerDetector(detOptions.harrisOptions_.value()));
      matcher_ = std::make_unique<OcvHarrisKeypointMatcher>(imageL_, imageR_);
    } else if (detOptions.detectorType_ == MPIHarrisDetector) {
      detector_ = std::make_unique<MPIHarrisCornerDetector>(
          detOptions.harrisOptions_.value(), options.pid_, options.nproc_);
      matcher_ = std::make_unique<MPIHarrisKeypointMatcher>(
          imageL_, imageR_, detOptions.harrisOptions_.value(), options.pid_,
          options.nproc_);
    } else {
      panic("Invalid detector type!");
    }

    if (options.ransacOptions_.ransacType_ == SeqRansac) {
      homographyCalculator_ =
          std::make_unique<SeqRansacHomographyCalculator>(ransacOptions);
    } else if (options.ransacOptions_.ransacType_ == OcvRansac) {
      homographyCalculator_ =
          std::make_unique<OcvRansacHomographyCalculator>(ransacOptions);
    }

    warpFunction_ = options.warpFunction_;
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
  // options
  PanoramicOptions options_;
  // feature detector
  std::unique_ptr<FeatureDetector> detector_;
  // matcher
  std::unique_ptr<KeyPointMatcher> matcher_;
  // homography calculator
  std::unique_ptr<RansacHomographyCalculator> homographyCalculator_;
  // warp function
  warpFunction_t warpFunction_;

  // we have two images to be stitched
  cv::Mat imageL_;
  cv::Mat imageR_;
};

#endif
