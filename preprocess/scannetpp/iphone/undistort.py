import cv2
import numpy as np
import os
import argparse

def read_camera_parameters(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Image dimensions
    width, height = map(float, lines[0].strip().split())
    
    # Camera intrinsic matrix
    K = np.array([list(map(float, lines[1].strip().split())),
                  list(map(float, lines[2].strip().split())),
                  list(map(float, lines[3].strip().split()))])
    
    # Distortion coefficients
    dist_coeffs = np.array(list(map(float, lines[4].strip().split())))
    
    return int(width), int(height), K, dist_coeffs

def undistort_image(image, K, dist_coeffs, width, height):
    # Get the optimal new camera matrix
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (width, height), 1, (width, height))
    
    # Undistort the image
    undistorted_img = cv2.undistort(image, K, dist_coeffs, None, K)
    
    return undistorted_img

def undistort_images_in_folder(folder_path, output_folder, camera_params_file):
    width, height, K, dist_coeffs = read_camera_parameters(camera_params_file)
    
    print(dist_coeffs)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue
            
            undistorted_img = undistort_image(image, K, dist_coeffs, width, height)
            
            # Save the undistorted image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, undistorted_img)
            print(f"Undistorted image saved: {output_path}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Undistort images using camera parameters.")
    parser.add_argument('images_folder', type=str, help="Path to the folder containing images to undistort.")
    parser.add_argument('output_folder', type=str, help="Path to the folder to save undistorted images.")
    parser.add_argument('camera_params_file', type=str, help="Path to the file containing camera parameters.")
    
    args = parser.parse_args()
    
    # Call the function to undistort images
    undistort_images_in_folder(args.images_folder, args.output_folder, args.camera_params_file)
