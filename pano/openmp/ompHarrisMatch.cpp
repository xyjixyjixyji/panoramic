#include <common.hpp>
#include <limits>
#include <matcher.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
#include <omp.h>

OmpHarrisKeypointMatcher::OmpHarrisKeypointMatcher(cv::Mat &image1,
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
OmpHarrisKeypointMatcher::matchKeyPoints(std::vector<cv::KeyPoint> keypointsL,
                                         std::vector<cv::KeyPoint> keypointsR) {
    const int num_threads = omp_get_max_threads();
    std::vector<std::vector<cv::DMatch>> perThreadMatches(num_threads);
    std::vector<cv::DMatch> allMatches;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int size = keypointsL.size();
        const int chunk_size = size / num_threads;
        const int start = tid * chunk_size;
        const int end = std::min(start + chunk_size, size);
        std::vector<cv::KeyPoint> localKeypointsL(keypointsL.begin() + start, keypointsL.begin() + end);
        std::vector<cv::DMatch> localMatches = seqHarrisMatchKeyPoints(
            localKeypointsL, keypointsR, image1_, image2_, options_, start);
        perThreadMatches[tid] = std::move(localMatches);
    }

    for (const auto& localMatches : perThreadMatches) {
        allMatches.insert(allMatches.end(), localMatches.begin(), localMatches.end());
    }

    return allMatches;
}
