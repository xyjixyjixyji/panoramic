# Panoramic

Parallel Panoramic Image Stitching

## Example Run

NOTE(FIXME): sometimes segfault will happen, just rerun the program

```shell
> make build
> ./build/pano_cmd --imgL ./data/viewL.png --imgR ./data/viewR.png
```

## Usage

```shell
> make build
> ./build/pano_cmd -h

Usage: Panoramic Image Stitcher [--help] [--version] --imgL VAR --imgR VAR [--detector VAR] [--harris-k VAR] [--harris-nms-thresh VAR] [--harris-nms-neigh VAR] [--harris-patch-size VAR] [--harris-max-ssd VAR] [--ransac-num-iter VAR] [--ransac-num-samples VAR] [--ransac-dist-thresh VAR]

Optional arguments:
-h, --help shows help message and exits
-v, --version prints version information and exits
--imgL The left image you want to stitch [required]
--imgR The right image you want to stitch [required]
--detector The type of feature detector to use: harris | ... [nargs=0..1] [default: "harris"]
--harris-k The k parameter for Harris Corner Detector [nargs=0..1] [default: 0.04]
--harris-nms-thresh The threshold for non-maximum suppression [nargs=0..1] [default: 300000]
--harris-nms-neigh The neighborhood size for non-maximum suppression [nargs=0..1] [default: 9]
--harris-patch-size The patch size for Harris Corner Detector [nargs=0..1] [default: 7]
--harris-max-ssd The maximum SSD between two patches we are okay with [nargs=0..1] [default: 2500]
--ransac-num-iter The number of iterations for RANSAC [nargs=0..1] [default: 500]
--ransac-num-samples The number of samples for each RANSAC iteration [nargs=0..1] [default: 4]
--ransac-dist-thresh The distance threshold for a point to be considered an inlier [nargs=0..1] [default: 5]
```
