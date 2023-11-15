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
   * @brief Stitch two images together to form a panorama
   *
   * @param image1 the first image
   * @param image2 the second image
   * @return the stitched panorama
   */
  cv::Mat stitch(const cv::Mat &image1, const cv::Mat &image2);

private:
  // feature detector
  std::unique_ptr<FeatureDetector> detector_;

  // TODO: matching function
  // TODO: get homographic transformation
  // TODO: warp the image
  // TODO: blend the image

  // we have two images to be stitched
  cv::Mat image1_;
  cv::Mat image2_;
};
