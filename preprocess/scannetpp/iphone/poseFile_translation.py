import numpy as np
import sys
import os

def quaternion_rotation_matrix(q0, q1, q2, q3):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
                  
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix


def transform_format(input_lines):
    transformed_data = []
    for line in input_lines:
        if line and not line.startswith('#') and not line.startswith('\n'):
            parts = line.split()
            image_id = int(parts[9].split("_")[-1].split(".")[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            # Hardcode some values as placeholders (adjust if necessary)
            index = 101  # Placeholder
            fixed_values = [0, 0, 0, 1]

            # Convert quaternion to rotation matrix
            rotation_matrix = quaternion_rotation_matrix(qw, qx, qy, qz)


            # Flatten the rotation matrix and combine with translation
            transformed_line = [image_id, index] + rotation_matrix[0].flatten().tolist() + [tx] + rotation_matrix[1].flatten().tolist() + [ty] + rotation_matrix[2].flatten().tolist() + [tz] + fixed_values
            transformed_data.append(" ".join(map(str, transformed_line)))

    return transformed_data

def main(input_file, output_file):
    with open(input_file, 'r') as f:
        input_lines = f.readlines()

    transformed_data = transform_format(input_lines)

    os.makedirs("/".join(output_file.split("/")[:-1]), exist_ok=True)

    with open(output_file, 'w') as f:
        for i, line in enumerate(transformed_data):
            f.write(line)
            if i != len(transformed_data) - 1:
                f.write("\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python poseFile_translation.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    main(input_file, output_file)
