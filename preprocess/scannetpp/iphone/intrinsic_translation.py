import sys

def translate_camera_file(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # Extract camera data from the lines
    camera_data = None
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            camera_data = line.split()
            break

    if camera_data is None:
        print("No camera data found.")
        return

    # Parse the camera parameters
    width = int(camera_data[2])  # WIDTH
    height = int(camera_data[3]) # HEIGHT
    fx = float(camera_data[4])
    fy = float(camera_data[5])
    cx = float(camera_data[6])
    cy = float(camera_data[7])
    distparams = list(map(float, camera_data[8:]))  # PARAMS

    # Prepare the output format
    output_lines = []
    output_lines.append(f"{width} {height}")
    output_lines.append(f"{fx} 0 {cx}")
    output_lines.append(f"0 {fy} {cy}")
    output_lines.append("0 0 1")  # Fixed last line as per example
    output_lines.append(' '.join(map(str, distparams)))  # Remaining parameters

    # Write to the output file
    with open(output_file, 'w') as outfile:
        for line in output_lines:
            outfile.write(line + '\n')

    print(f"Translated camera data written to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python intrinsic_translation.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    translate_camera_file(input_file, output_file)
