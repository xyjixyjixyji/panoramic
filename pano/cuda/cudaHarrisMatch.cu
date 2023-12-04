#include <common.hpp>
#include <limits>
#include <matcher.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

CudaHarrisKeypointMatcher::CudaHarrisKeypointMatcher(cv::Mat &image1,
                                                   cv::Mat &image2,
                                                   HarrisCornerOptions options)
    : image1_(image1), image2_(image2), options_(options) {}

/**
 * @brief Match keypoints detected by Harris corner detector
 *
 * We typically take a patch around the keypoint and compare the distance
 *
 * @return std::vector<cv::DMatch> the matches
 */
std::vector<cv::DMatch>
CudaHarrisKeypointMatcher::matchKeyPoints(std::vector<cv::KeyPoint> keypointsL,
                                         std::vector<cv::KeyPoint> keypointsR) {
  // options
  const int patchSize = options_.patchSize_;
  const double maxSSDThresh = options_.maxSSDThresh_;

  std::vector<cv::DMatch> matches;
  int border = patchSize / 2;

  for (size_t i = 0; i < keypointsL.size(); i++) {
    const auto &kp1 = keypointsL[i];
    cv::Point2f pos1 = kp1.pt;

    if (pos1.x < border || pos1.y < border || pos1.x + border >= image1_.cols ||
        pos1.y + border >= image1_.rows) {
      continue;
    }

    int bestMatchIndex = -1;
    double bestMatchSSD = std::numeric_limits<double>::max();
    for (size_t j = 0; j < keypointsR.size(); j++) {
      const auto &kp2 = keypointsR[j];
      cv::Point2f pos2 = kp2.pt;

      if (pos2.x < border || pos2.y < border ||
          pos2.x + border >= image2_.cols || pos2.y + border >= image2_.rows) {
        continue;
      }

      double ssd = 0;
      for (int y = -border; y <= border; y++) {
        for (int x = -border; x <= border; x++) {
          cv::Vec3b p1 = image1_.at<cv::Vec3b>(pos1.y + y, pos1.x + x);
          cv::Vec3b p2 = image2_.at<cv::Vec3b>(pos2.y + y, pos2.x + x);
          double diff = 0;
          for (int c = 0; c < 3; c++) {
            diff += (p1[c] - p2[c]) * (p1[c] - p2[c]);
          }
          ssd += pow(diff, 2);
        }
      }

      if (ssd < bestMatchSSD) {
        bestMatchSSD = ssd;
        bestMatchIndex = j;
      }
    }

    if (bestMatchSSD < maxSSDThresh) {
      matches.push_back(cv::DMatch(i, bestMatchIndex, bestMatchSSD));
    }
  }

  return matches;
}
