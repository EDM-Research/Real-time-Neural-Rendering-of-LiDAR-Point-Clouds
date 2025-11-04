# Real-time Neural Rendering of LiDAR Point Clouds

This is the official implementation of the preseted short paper at Eurographics 2025: [Real-time Neural Rendering of LiDAR Point Clouds](https://doi.org/10.2312/egs.20251041).

*Static LiDAR scanners produce accurate, dense, colored point clouds, but often contain obtrusive artifacts which makes them ill-suited for direct display. We propose an efficient method to render more perceptually realistic images of such scans without any expensive preprocessing or training of a scene-specific model. A naive projection of the point cloud to the output view using 1×1 pixels is fast and retains the available detail, but also results in unintelligible renderings as background points leak between the foreground pixels. The key insight is that these projections can be transformed into a more realistic result using a deep convolutional model in the form of a U-Net, and a depth-based heuristic that prefilters the data. The U-Net also handles LiDAR-specific problems such as missing parts due to occlusion, color inconsistencies and varying point densities. We also describe a method to generate synthetic training data to deal with imperfectly-aligned ground truth images. Our method achieves real-time rendering rates using an off-the-shelf GPU and outperforms the state-of-the-art in both speed and quality.*

## Citation

```
@inproceedings{10.2312:egs.20251041,
    booktitle = {Eurographics 2025 - Short Papers},
    editor = {Ceylan, Duygu and Li, Tzu-Mao},
    title = {{Real-time Neural Rendering of LiDAR Point Clouds}},
    author = {VANHERCK, Joni and Zoomers, Brent and Mertens, Tom and Jorissen, Lode and Michiels, Nick},
    year = {2025},
    publisher = {The Eurographics Association},
    ISSN = {1017-4656},
    ISBN = {978-3-03868-268-4},
    DOI = {10.2312/egs.20251041}
}
```

## Installation - LINUX

#### CMake

https://cmake.org/download/

#### OpenCV

https://github.com/opencv/opencv

#### Torch Tensor RT support

- Download libtorchtrt-* from https://github.com/pytorch/TensorRT/releases and look at the latest release and the version of the dependencies. 
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

See `example` for usage. Scenes can be downloaded from [ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/).
Please download the pre-trained model using Git LFS, or manually download the files if Git LFS is not available.

To compile the model for your architecture using Tensor RT do the following:
```
python -m pip install torch torch-tensorrt tensorrt torchvision --extra-index-url https://download.pytorch.org/whl/cuxxx <-- fill in your cuda version
```
Then change the export_ts.py file to compile for your specific resolution(s), and run `example`.

Without Tensor RT support run the export_pt.py file, uncomment `line 77` and run `example`.


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


## About
![](/img/DFL_FlandersMake.jpg)

This work is a development by [Hasselt University](https://www.uhasselt.be/), [Digital Future Lab](https://www.uhasselt.be/en/instituten-en/digitalfuturelab), funded by Hasselt University.
