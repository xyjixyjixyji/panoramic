#include <common.hpp>
#include <detector.hpp>
#include <omp.h>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

const int ROWS_PER_CHUNK = 10;

OmpHarrisCornerDetector::OmpHarrisCornerDetector(HarrisCornerOptions options)
    : options_(options) {}

/**
 * @brief Detect keypoints in the input image by Harris corner method
 *
 * @param image input image is rather a grayscale image or a BGR image
 * @return std::vector<cv::KeyPoint> the keypoints detected
 */
std::vector<cv::KeyPoint>
OmpHarrisCornerDetector::detect(const cv::Mat &image) {
    std::vector<cv::KeyPoint> allKeypoints;

    #pragma omp for nowait
    for (int i = 0; i < image.rows; i += ROWS_PER_CHUNK) {
        cv::Mat subImage = image(cv::Range(i, std::min(image.rows, i+ROWS_PER_CHUNK)), 
                                cv::Range::all());
        std::vector<cv::KeyPoint> keypoints = seqHarrisCornerDetectorDetect(
            subImage, options_);
        for (auto &keypoint : keypoints) {
            keypoint.pt.y += i;
        }

        #pragma omp critical
        {
            allKeypoints.insert(allKeypoints.end(), 
                keypoints.begin(), 
                keypoints.end());
        }
    }

    return allKeypoints;
}