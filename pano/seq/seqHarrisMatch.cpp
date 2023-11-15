#include <common.hpp>
#include <limits>
#include <matcher.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

const int patchSize = 7;
const double maxSSDThresh = 2500;

/**
 * @brief Match keypoints detected by Harris corner detector
 *
 * We typically take a patch around the keypoint and compare the distance
 *
 * @return std::vector<cv::DMatch> the matches
 */
std::vector<cv::DMatch>
SeqHarrisKeyPointMatcher::matchKeyPoints(std::vector<cv::KeyPoint> keypointsL,
                                         std::vector<cv::KeyPoint> keypointsR) {
  std::vector<cv::DMatch> matches;
  int border = patchSize / 2;

  for (size_t i = 0; i < keypointsL.size(); i++) {
    const auto &kp1 = keypointsL[i];
    cv::Point2f pos1 = kp1.pt;

    if (pos1.x < border || pos1.y < border || pos1.x + border >= image1_.cols ||
        pos1.y + border >= image1_.rows) {
      continue;
    }

    cv::Mat patch1 = image1_(
        cv::Rect(pos1.x - border, pos1.y - border, patchSize, patchSize));

    size_t bestMatchIndex = -1;
    double bestMatchSSD = std::numeric_limits<double>::max();
    for (size_t j = 0; j < keypointsR.size(); j++) {
      const auto &kp2 = keypointsR[j];
      cv::Point2f pos2 = kp2.pt;

      if (pos2.x < border || pos2.y < border ||
          pos2.x + border >= image2_.cols || pos2.y + border >= image2_.rows) {
        continue;
      }

      cv::Mat patch2 = image2_(
          cv::Rect(pos2.x - border, pos2.y - border, patchSize, patchSize));

      double ssd = computeSSD(patch1, patch2);
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
