# ScanNet++ preprocessing

## Requirements

Install ffmpeg from https://www.ffmpeg.org/

```
conda create -y -n scannetpp python=3.10
conda activate scannetpp
pip install -r requirements.txt
```

## Usage

- Download ScanNet++: https://kaldir.vc.in.tum.de/scannetpp/
    - Make sure to include point clouds and iphone data
- Change the `iphone/prepare_iphone_data.yml` to use the correct folder and scene_ids
- Change `saving_folder` in `iphone/prepare_iphone_data.py`
- Run: 

```
python -m iphone.prepare_iphone_data ./iphone/prepare_iphone_data.yml
```

Output is a per-scene folder with undistorted iphone images, camera intrinsics file, binary point cloud, octree point cloud (.oct) and a trajectory file.