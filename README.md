# Real-time Neural Rendering of LiDAR Point Clouds

## Installation - LINUX

#### CMake

https://cmake.org/download/

#### OpenCV

https://github.com/opencv/opencv

#### Torch Tensor RT support
- Go to https://github.com/pytorch/TensorRT/releases and look at the latest release and the dependencies. 
- **Make sure to install the correct versions of the following dependencies**: libtorch, TensorRT, CUDA
- Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
- Install libtorch (https://pytorch.org/get-started/locally/) and torch tensor RT precompiled binaries to lib/ folder. 
- Download Tensor RT from https://developer.nvidia.com/tensorrt/download/ and install it using these commands:
```
sudo dpkg -i nv-tensorrt-repo-*.deb
<execute the last line of dpkg output>
sudo apt update
sudo apt install tensorrt
```

#### No Tensor RT support
- Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
- Install libtorch (https://pytorch.org/get-started/locally/) precompiled binaries to lib/ folder. 

#### LibE57Format

Install libE57Format from https://github.com/asmaloney/libE57Format

### Build
```
cd src
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4 && sudo make install
```

## Usage 

### Preprocessing

#### ScanNet++ scenes

- Follow preprocess/scannetpp README
