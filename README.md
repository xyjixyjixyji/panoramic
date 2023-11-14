Members: Xinyou Ji(xinyouj), Zihe Zhao(zihezhao)

[Project URL](https://github.com/Ji-Xinyou/panoramic)

---

## Summary
Large-scale panoramic image creation, also known as image stitching, is a process to create one expansive picture by combining multiple individual images with overlapping regions. The goal of this project is to implement this expensive task in C++ and parallelize it with CUDA and OpenMP.

## Backgroud
<p align="center">
  <img width="625" alt="image" src="https://github.com/Ji-Xinyou/panoramic/assets/70172199/0eee70d3-0d62-415f-b1df-80bb08976b83">
</p>
<p align="center">
Fig 1. Example and process of image stitching. Source: https://yungyung7654321.medium.com.
</p>

Figure 1 shows the pipeline and an example of stitching images together. On a high level, we find feature key points in each image. Then, we match features across neighboring images. Finally, we combine them based on the matched feature points. In the following subsections, we further discuss each step and how we plan to approach the parallelization.

### Image acuisition
In this step, we simply collect a series of images with overlapping regions to be stitched together. This step is merely data acquisition and thus does not need parallel processing.

### Identify feature

In this step, we identify feature key points in each image. Feature key points are the points (*e.g.* corners) that are invariant even if an image shifts or rotates. There are several methods for detecting feature key points. Some popular choices in the C++ OpenCV library[^1^] are (ordered by year of publication):

- **Harris Corner Detection**[^2^] (`cv::cornerHarris`): a traditional approach published in 1988.
- **Scale-Invariant Feature Transform (SIFT)**[^3^] (`cv::xfeatures2d::SIFT`): a method robust to scale and rotation changes that is based on the local gradient around a key-point.
- **Speeded-Up Robust Features (SURF)**[^4^] (`cv::xfeatures2d::SURF`): an accelerated method akin to SIFT, leveraging integral images for better efficiency.
- **Oriented FAST and Rotated BRIEF (ORB)**[^5^] (`cv::ORB`): a method designed to have efficiency and good performance in real-time applications.

We will explore more and determine the one that fits most with our project. In the example in Figure 1, only key points in the overlapping region were shown. However, in actual implementation, we need to find all key points in an entire image for all images -- we could not directly define an overlapping region for all neighboring images; instead we rely on feature matching to find the overlapping region. We plan to use CUDA to parallelize this step.

### Feature matching
For each pair of neighboring images, we use feature matching to determine if there is an overlapping region. Specifcally, we use feature descriptors to calculate the features gradient around each feature key point and then compare every pair of points across the two images. We mark the pairs of points with minimized Euclidean distances as the overlapping region. We can implement this step with either CUDA or OpenMP.

### Blend images
<p align="center">
  <img width="300" alt="image" src="https://github.com/Ji-Xinyou/panoramic/assets/70172199/39d5a2ec-e5c1-43da-baeb-105bb2a0084f">
</p>
<p align="center">
Fig 2. Example of homography transformation[^6^].
</p>

Homography involves a matrix transformation, as demonstrated in Figure 2, that can describe the projective (*e.g.* affine and rotation) translation from one set of points to another set of points. In our task, after finding the matching key points, we will need to find the homography matrix that can translate an image. To reduce the impact of falsely matched points, we use the Random Sample Consensus (RANSAC) [^7^], an algorithm that finds the best homography transformation by:

1. Iteratively selecting subsets of corresponding points.
2. Estimating a homography for each subset.
3. Selecting the model with the highest inlier count.

We might be able to parallelize this step through parallelizing the calculation for different subsets or the homography estimation step itself.

## The Challenge

## Resources
We are planning to write the code from scratch. However, C++ OpenCV library provides the implementation of many components in our project *e.g.* feature detector. We plan to seek online code base for the sequential implementation and will cite any referenced code base in our reports.

## Goals and Deliverables
### Plan to and hope to achieve
We plan to achieve acceptable panoramic image creation -- we will test this with both real-life images and OpenCV generated data. We also plan to deploy the final project on our website so that users can upload images and have them merged into one panoramic photo. If we have time, we hope to further refine the image blending step such as adding color correction.

### Demo
We plan to demo the actual panoramic photo creation functionality deployed on our website. It might be an interactive demo as we might take the input pictures on the spot. We will demonstrate speedup graphs to show that our parallelization is effective.

### Analysis / systems project only
Not applicable.

## Platform Choice
We plan to use C++ with parallelization achieved primarily by CUDA but maybe also OpenMP. We will be mostly using the GHC machines but might also use PSC for performance benchmarking. 

## Schedule
| Date          | Week | Content                                        |
|---------------|------|------------------------------------------------|
| 11/19-11/25   | 1    | Implement sequential version and design parallelization.|
| 11/26-12/2    | 2    | Implement parallelization.                     |
| 12/3-12/9     | 3    | Optimization and benchmark.                     |
| 12/10-12/15   | 4    | Writeup and presentation.                       |

[^1^]: [OpenCV library](https://opencv.org/)
[^2^]: Harris, C., & Stephens, M. (1988). A Combined Corner and Edge Detector. In *Proceedings of the Alvey Vision Conference*.
[^3^]: Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. *International Journal of Computer Vision, 60*(2), 91–110.
[^4^]: Bay, H., Ess, A., Tuytelaars, T., & Van Gool, L. (2008). Speeded-Up Robust Features (SURF). *Computer Vision and Image Understanding (CVIU), 110*(3), 346–359.
[^5^]: Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). ORB: An efficient alternative to SIFT or SURF. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.
[^6^]: Lai, Po-Lun (Ryan) & Yilmaz, Alper. (2023). Projective reconstruction of building shape from silhouette images acquired from uncalibrated cameras.
[^7^]: Fischler, M. A., & Bolles, R. C. (1981). Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography. *Communications of the ACM (CACM), 24*(6), 381–395.
