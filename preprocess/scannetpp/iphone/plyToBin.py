import open3d as o3d
import numpy as np
import sys

def ply_to_bin(ply_file, bin_file):
    # Load the PLY file using Open3D
    pcd = o3d.io.read_point_cloud(ply_file)

    # o3d.visualization.draw_geometries([pcd], window_name='Open3D - Point Cloud Viewer')

    
    # Convert the point cloud data to numpy arrays
    points = np.asarray(pcd.points)  # XYZ coordinates as float32
    colors = np.asarray(pcd.colors)  # RGB values normalized between 0 and 1

    # Ensure that RGB values are in the range [0, 255] as uint8
    colors = (colors * 255).astype(np.uint8)

    colors = colors[:, [2, 1, 0]]  # Swap columns to convert RGB to BGR


    # Get the number of points
    num_points = points.shape[0]
    
    # Open the binary file for writing
    with open(bin_file, 'wb') as f:
        # Write the number of points (size_t)
        f.write(np.array([num_points], dtype=np.uint64).tobytes())  # Write as 64-bit unsigned integer
        
        # Write the point cloud data (XYZ as float32)
        f.write(points.astype(np.float32).tobytes())
        
        # Write the color data (RGB as uint8)
        f.write(colors.astype(np.uint8).tobytes())

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ply_to_bin.py <input.ply> <output.bin>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    bin_file = sys.argv[2]
    
    ply_to_bin(ply_file, bin_file)
