import os

def validate_folder_structure(root_folder):
    # Define the expected folder structure
    expected_structure = [
        "960_0.1/ours",
        "960_0.1/pointersect",
        "960_0.4/ours",
        "960_0.4/pointersect",
        "960_0.7/ours",
        "960_0.7/pointersect",
        "960_1/ours",
        "960_1/pointersect",
        "1920_0.1/ours",
        "1920_0.1/pointersect",
        "1920_0.4/ours",
        "1920_0.4/pointersect",
        "1920_0.7/ours",
        "1920_0.7/pointersect",
        "1920_1/ours",
        "1920_1/pointersect",
    ]

    # Iterate over each subfolder in the root folder
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        # Skip if it's not a directory
        if not os.path.isdir(subfolder_path):
            continue

        print(f"Checking structure for: {subfolder}")
        
        # Check each expected path
        for path in expected_structure:
            # Replace the root folder with the current subfolder path
            target_path = os.path.join(subfolder_path, path)

            # Check if the path exists
            if not os.path.exists(target_path):
                print(f"  Missing: {os.path.relpath(target_path, subfolder_path)}")

if __name__ == "__main__":
    folder_to_check = "D:/Unseen_v2/"
    if os.path.exists(folder_to_check) and os.path.isdir(folder_to_check):
        validate_folder_structure(folder_to_check)
    else:
        print("The specified folder does not exist or is not a directory.")
