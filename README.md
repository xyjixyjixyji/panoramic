# Panoramic

Parallel Panoramic Image Stitching

## Example Run

! NOTE: opencv only supports two image stitching, our mpi/omp/cuda/seq supports any number of image stitching
! NOTE: on macOS, run `export TMPDIR=/tmp` to avoid unlink syscall failure

```shell
# For regular runs
> make build
> ./build/pano_cmd --detector seq --ransac ocv --warp ocv --img ./data/viewL.png --img ./data/viewR.png

# For multiple images
> ./build/pano_cmd --detector seq --ransac ocv --warp ocv --img ./data/v1.png --img ./data/v2.png --img ./data/v3.png --img ./data/v4.png

# For mpi runs
> mpirun -n 1 ./build/pano_cmd --detector mpi --ransac ocv --warp ocv --img ./data/viewL.png --img ./data/viewR.png
```

## Usage

```shell
> make build
> ./build/pano_cmd -h

Usage: Panoramic Image Stitcher [--help] [--version] --img VAR [--detector VAR] [--ransac VAR] [--warp VAR] [--harris-k VAR] [--harris-nms-thresh VAR] [--harris-nms-neigh VAR] [--harris-patch-size VAR] [--harris-max-ssd VAR] [--ransac-num-iter VAR] [--ransac-num-samples VAR] [--ransac-dist-thresh VAR]

Optional arguments:
  -h, --help            shows help message and exits
  -v, --version         prints version information and exits
  --img                 The images you want to stitch, from **left to right** [required]
  --detector            The type of feature detector to use: seq | ocv | mpi | ... [nargs=0..1] [default: "seq"]
  --ransac              The type of RANSAC to use: seq | ocv [nargs=0..1] [default: "seq"]
  --warp                The type of warp function to use: seq | ocv | ... [nargs=0..1] [default: "seq"]
  --harris-k            The k parameter for Harris Corner Detector [nargs=0..1] [default: 0.03]
  --harris-nms-thresh   The threshold for non-maximum suppression [nargs=0..1] [default: 5000]
  --harris-nms-neigh    The neighborhood size for non-maximum suppression [nargs=0..1] [default: 3]
  --harris-patch-size   The patch size for Harris Corner Detector [nargs=0..1] [default: 5]
  --harris-max-ssd      The maximum SSD between two patches we are okay with [nargs=0..1] [default: 2500]
  --ransac-num-iter     The number of iterations for RANSAC [nargs=0..1] [default: 1000]
  --ransac-num-samples  The number of samples for each RANSAC iteration [nargs=0..1] [default: 4]
  --ransac-dist-thresh  The distance threshold for a point to be considered an inlier [nargs=0..1] [default: 5]
```
