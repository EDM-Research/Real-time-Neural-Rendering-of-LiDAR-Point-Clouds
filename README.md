# Real-time Neural Rendering of LiDAR Point Clouds

## Installation - LINUX

#### CMake

https://cmake.org/download/

#### OpenCV

https://github.com/opencv/opencv

#### Torch Tensor RT support
- Download torch-tensorrt from https://github.com/pytorch/TensorRT/releases and look at the latest release and the dependencies. 
- **Make sure to install the correct versions of the following dependencies**: libtorch, TensorRT, CUDA
- Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
- Install libtorch (https://pytorch.org/get-started/locally/) and torch tensor RT precompiled binaries to /usr/local folder. 
- Download Tensor RT from https://developer.nvidia.com/tensorrt/download/ and install it using these commands:
```
sudo dpkg -i nv-tensorrt-local-repo-*.deb
<execute the last line of dpkg output>
sudo apt update
sudo apt install tensorrt
```
Make sure the correct version is installed. If you have an nvidia repository, the latest available version of tensorrt could be pulled from there. 


#### No Tensor RT support
- Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
- Install libtorch (https://pytorch.org/get-started/locally/) precompiled binaries to /usr/local folder. 

#### LibE57Format

Install libE57Format from https://github.com/asmaloney/libE57Format

### Build
RTRenderer lib
```
cd src
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4 && sudo make install
```

Example code
```
cd example
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4 && sudo make install
```

## Usage 

See example for example code. 

To compile the model for your architecture using Tensor RT do the following:
```
python -m pip install torch torch-tensorrt tensorrt torchvision --extra-index-url https://download.pytorch.org/whl/cuxxx <-- fill in your cuda version
```
Then change the export_ts.py file to compile for your specific resolution(s), and run `example` using the exported .ts file on `line 73`. 

```
render_trajectory <path_to_point_cloud> <path_to_trajectory> <path_to_intrinsics>
```
- Point cloud file should be .e57 or .ply
- Trajectory in format: `IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME` (output of COLMAP)
- Intrinsics .txt file: 
    - if named cameras.txt, COLMAP output is expected: `CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]`
    - in other cases, this format is expected:
```
        W H
        fx 0 cx
        0 fy cy
        0 0 1
        k1 k2 p1 p2 k3
        0
```
