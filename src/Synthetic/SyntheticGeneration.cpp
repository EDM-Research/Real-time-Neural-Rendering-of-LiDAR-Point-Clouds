#include "RTRenderer/cloudreader.h"
#include "RTRenderer/project_cloud.h"


int main(int argc, char* argv[])
{
    if(argc < 3)
    {
        std::cerr << "Not enough arguments given: synthetic_generation <folder> <output_folder>" << std::endl;
    }

    std::filesystem::path folder = argv[1];
    std::filesystem::path folder_output = argv[2];

    for (const auto& entry : std::filesystem::directory_iterator(folder))
    {
        std::filesystem::path plyFile = folder / entry.path().stem() /  "scans" / "pc_aligned.ply";

        auto grid = CloudReader::loadCloud(plyFile);

        // todo read intrinsics file and create camera calibration
        // read trajectory file and render poses with ProjectCloud::computeRGBD()
        // read image for each pose
        // use depth as a mask for original image
        // apply noise variations from Utils
        // save img to correct output folder with utils::writeimg

    }


    return 0;
}



/*template <typename T>
std::pair<std::vector<T>, std::vector<T>> splitVector(const std::vector<T>& input, float ratio) {
	// Validate that ratio is between 0 and 1
	if (ratio < 0.0 || ratio > 1.0) {
		throw std::invalid_argument("Ratio must be between 0 and 1.");
	}

	// Copy input vector and shuffle it
	std::vector<T> shuffled(input);
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(shuffled.begin(), shuffled.end(), g);

	// Calculate sizes for the split
	size_t splitIndex = std::round(ratio * shuffled.size());

	// Create the two resulting vectors
	std::vector<T> firstPart(shuffled.begin(), shuffled.begin() + splitIndex);
	std::vector<T> secondPart(shuffled.begin() + splitIndex, shuffled.end());

	return std::make_pair(firstPart, secondPart);
}

void processVector(const std::vector<std::pair<int, cv::Matx44d>>& poses, std::atomic_int& counter, int totalPosesSize, const CameraCalibration& calibration, const std::unordered_map<int, OctreeGrid::Block>& grid,
					const std::string& output_path, const std::string& input_path, const std::string& folder, const std::string& split)
{
	#pragma omp parallel for
	for (int i = 0; i < poses.size(); ++i) {
		std::pair<int, cv::Matx44d> keyframepose = poses[i];

		std::cout << "[" << int((float(counter + 1) / totalPosesSize) * 100.0f) << "%]\t";

		cv::Size imageSize(calibration.getWidth(), calibration.getHeight());
		std::stringstream ssInput;
		ssInput << input_path + "/frame_" << std::setw(6) << std::setfill('0') << keyframepose.first << ".jpg";
		cv::Mat inputImg, outputImg;
		Utils::processImage(grid, imageSize, keyframepose.second, calibration.getIntrinsicsMatrix(), ssInput.str(), inputImg, outputImg, false);

		Utils::writeToFile(output_path, inputImg, outputImg, split, folder, keyframepose.first);

		std::cout << "Saved frame " << keyframepose.first << std::endl;
		counter++;
	}
}


int main(int argc, char* argv[])
{
	srand(static_cast<unsigned int>(time(0)));

	CmdLineParser cml(argc, argv);
	std::cout << "-folder : " << (cml["-folder"] ? cml("-folder") : "") << std::endl;
	std::cout << "-output : " << (cml["-output"] ? cml("-output") : "") << std::endl;

	float training_split_ratio = 0.8;

	for (const auto& entry : std::filesystem::directory_iterator(cml("-folder"))) {
		if (entry.is_directory()) {
			std::string folder = entry.path().stem().string();
			std::cout << folder << std::endl;
			std::string output_path = cml("-output");
			std::string input_path = cml("-folder") + folder + "/imgs/";
			std::string keyFramePoses_file = cml("-folder") + folder + "/trajectory.txt";
			std::string pointcloud_oct_file = cml("-folder") + folder + "/pcd.oct";
			std::string intrinsics_file = cml("-folder") + folder + "/camera_intrinsics.txt";

			std::cout << "SETTINGS: " << std::endl;
			std::cout << "\t" << "Rendering poses of: " << keyFramePoses_file << std::endl;
			std::cout << "\t" << "Output files will be saved to path: " << output_path << " with input_path " << input_path << std::endl;
			std::cout << "\t" << "Point cloud to render: " << pointcloud_oct_file << std::endl;
			std::cout << "\t" << "Intrinsics file: " << intrinsics_file << std::endl;

			std::cout << "Creating folder \"" << output_path + "/\"" << std::endl;
			if (!std::filesystem::is_directory(output_path) || !std::filesystem::exists(output_path)) { // Check if src folder exists
				std::filesystem::create_directories(output_path + "/train_input"); // create src folder
				std::filesystem::create_directories(output_path + "/train_output"); // create src folder
				std::filesystem::create_directories(output_path + "/val_input"); // create src folder
				std::filesystem::create_directories(output_path + "/val_output"); // create src folder
			}

			CameraCalibration calibration;
			if (!calibration.loadCalibration(intrinsics_file)) {
				std::cerr << "Failed to open intrinsics file: " << intrinsics_file << std::endl;
				return -1;
			};

			cv::Matx33d newIntrinsic = calibration.getIntrinsicsMatrix();
			newIntrinsic = newIntrinsic / Config::resizeParam;
			newIntrinsic(2,2) = 1;
			calibration.setIntrinsicsMatrix(newIntrinsic);
			calibration.setHeight(static_cast<int>(calibration.getHeight() / Config::resizeParam));
			calibration.setWidth(static_cast<int>(calibration.getWidth() / Config::resizeParam));


			std::cout << "calib: " << calibration.getIntrinsicsMatrix() << std::endl;
			std::ifstream f;
			f.open(keyFramePoses_file.c_str());

			float timestamp;

			std::vector<std::pair<int, cv::Matx44d>> keyframePoses;
			while (!f.eof()) {
				int id;
				f >> id;
				f >> timestamp;
				//std::cout << timestamp << std::endl;
				cv::Matx44d cvPose;
				for (int row = 0; row < 4; ++row)
					for (int col = 0; col < 4; ++col)
						f >> cvPose(row, col);
				keyframePoses.emplace_back(std::pair<int, cv::Matx44d>(id, cvPose));
				//std::cout << cvPose << std::endl;
			}

			// octree data
			std::unordered_map<int, OctreeGrid::Block> grid;

			// load octree
			int numBlocks_x, numBlocks_y, numBlocks_z;
			OctreeGrid::readOctreeBinary(pointcloud_oct_file, grid, numBlocks_x, numBlocks_y, numBlocks_z);

			OctreeGrid::downSample(grid, Config::decimateParam);

			std::cout << "Loaded octree of dimensions: (" << numBlocks_x << ", " << numBlocks_y << ", " << numBlocks_z << ") and " << grid.size() << " blocks." << std::endl;
			std::atomic_int counter = 0;

			auto trainValVectors = splitVector(keyframePoses, training_split_ratio);

			processVector(trainValVectors.first, counter, keyframePoses.size(), calibration, grid, output_path, input_path, folder, "train");
			processVector(trainValVectors.second, counter, keyframePoses.size(), calibration, grid, output_path, input_path, folder, "val");
		}
	}


	return 0;
}
*/
