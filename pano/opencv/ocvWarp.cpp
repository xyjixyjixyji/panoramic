#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat warpOcv(cv::Mat imageL, cv::Mat imageR, cv::Mat homography) {
  cv::Mat warped;
  cv::warpPerspective(imageL, warped, homography,
                      imageL.size() + imageR.size());
  cv::Mat half(warped, cv::Rect(0, 0, imageR.cols, imageR.rows));
  imageR.copyTo(half);
  return warped;
}
