#include <fstream>
#include <unordered_map>
#include "../common/CameraCalibration.h"
#include "../common/Octreegrid.h"
#include "project_cloud.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#ifdef WIN32
#include <Windows.h>
#endif

std::vector<cv::Matx44d> interpolatePoses(const cv::Matx44d& pose1, const cv::Matx44d& pose2, int numInterpolations) {
	std::vector<cv::Matx44d> interpolatedPoses;

	cv::Matx33d R1(pose1.get_minor<3, 3>(0, 0));
	cv::Vec3d t1(pose1(0, 3), pose1(1, 3), pose1(2, 3));

	cv::Matx33d R2(pose2.get_minor<3, 3>(0, 0));
	cv::Vec3d t2(pose2(0, 3), pose2(1, 3), pose2(2, 3));

	cv::Vec3d rvec1, rvec2;
	cv::Rodrigues(cv::Mat(R1), rvec1);
	cv::Rodrigues(cv::Mat(R2), rvec2);

	for (int i = 0; i <= numInterpolations; ++i) {
		double alpha = static_cast<double>(i) / numInterpolations;

		cv::Vec3d rvec_interp = (1 - alpha) * rvec1 + alpha * rvec2;
		cv::Matx33d R_interp;
		cv::Rodrigues(rvec_interp, R_interp);

		cv::Vec3d t_interp = (1 - alpha) * t1 + alpha * t2;

		cv::Matx44d pose = cv::Matx44d::eye();
		for (int row = 0; row < 3; ++row)
			for (int col = 0; col < 3; ++col)
				pose(row, col) = R_interp(row, col);

		pose(0, 3) = t_interp[0];
		pose(1, 3) = t_interp[1];
		pose(2, 3) = t_interp[2];

		interpolatedPoses.push_back(pose);
	}

	return interpolatedPoses;
}

std::vector<int> linspace(int start, int end, int num) {
	if (num <= 0) {
		throw std::invalid_argument("Number of samples must be greater than 0");
	}

	std::vector<int> result;
	if (num == 1) {
		result.push_back(start);
		return result;
	}

	double step = (double)(end - start) / (double)(num - 1);

	for (int i = 0; i < num; ++i) {
		result.push_back(std::ceil(start + i * step));
	}


	return result;
}

std::vector<std::pair<int, cv::Matx44d>> readTrajectory(std::string filename)
{
	std::ifstream f;
	f.open(filename.c_str());
	float timestamp;
	std::vector<std::pair<int, cv::Matx44d>> keyframePoses;
	while (!f.eof()) {
		int id;
		f >> id;
		f >> timestamp;
		cv::Matx44d cvPose;
		for (int row = 0; row < 4; ++row)
			for (int col = 0; col < 4; ++col)
				f >> cvPose(row, col);
		keyframePoses.push_back(std::make_pair(id, cvPose));
	}

	return keyframePoses;

	// get 5 linspace poses, same implementation as np.linspace
	std::vector<int> ids = linspace(0, keyframePoses.size()-1, 5);

	std::vector<std::pair<int, cv::Matx44d>> resultPoses;
	for (int idx : ids)
	{
		if (idx >= 0 && idx < static_cast<int>(keyframePoses.size()))
		{
			resultPoses.push_back(keyframePoses[idx]);
		}
	}

	return resultPoses;
}

int main(int argc, char** argv)
{
	if (argc < 5)
	{
		std::cerr << "Not enough arguments: baseFolder sceneId downsampleRatio resolutionDownsample\n\n";
		return 0;
	}
#ifdef WIN32
    HMODULE loadLib = LoadLibraryA("torch_cuda.dll");
    if (loadLib == NULL) std::cout << "Failed to load torch_cuda.dll.\n";
#endif
	std::string baseFolder = argv[1]; //"C:/Users/JVanherck/PCLRendering/PCR/data/";

	std::string sceneId = argv[2]; //"3db0a1c8f3";

	float downsampleRatio = std::stof(argv[3]);
	float resolutionDownsample = std::stof(argv[4]);

	// read calibration file
	std::string intrinsics_file = baseFolder + sceneId + "/camera_intrinsics.txt";
	CameraCalibration calibration;
	if (!calibration.loadCalibration(intrinsics_file)) {
		std::cerr << "Failed to open intrinsics file: " << intrinsics_file << std::endl;
		return -1;
	};

	if (resolutionDownsample > 1)
	{
		calibration = calibration.getScaledCalibration(calibration.getWidth() / resolutionDownsample, calibration.getHeight() / resolutionDownsample);
	}

	cv::Matx33d intrinsics = calibration.getIntrinsicsMatrix();

	// read trajectory file
	std::vector<std::pair<int, cv::Matx44d>> keyframeVector = readTrajectory(baseFolder + sceneId + "/trajectory.txt");

	// read octree
	std::string pointcloud_oct_file = baseFolder + sceneId + "/pcd.oct";
	std::unordered_map<int, OctreeGrid::Block> grid;
	int numBlocks_x, numBlocks_y, numBlocks_z;
	OctreeGrid::readOctreeBinary(pointcloud_oct_file, grid, numBlocks_x, numBlocks_y, numBlocks_z);

	if (downsampleRatio < 1)
	{
		OctreeGrid::downSample(grid, downsampleRatio);
		OctreeGrid::writeOctreeBinary(baseFolder + sceneId + "/down.oct", grid, numBlocks_x, numBlocks_y, numBlocks_z);
	}

	// get camera pose and image size
	cv::Size imageSize = cv::Size(calibration.getWidth(), calibration.getHeight());

	ProjectCloud projector{ grid, calibration, baseFolder + "model.pt" };

	int maxDuration = 0;
	int maxBlocks = 0;
	int sumDuration = 0;

	// video 
    std::string outputFile = baseFolder + sceneId + "/video.mkv";
	cv::VideoWriter writer(outputFile, cv::VideoWriter::fourcc('H', '2', '6', '4'), 26, imageSize);


	std::stringstream saveFolder;
	saveFolder << baseFolder << sceneId << "/" << calibration.getWidth() << "_" << downsampleRatio << "/ours/batch_0";
	if (!std::filesystem::exists(saveFolder.str()))
	{
		std::filesystem::create_directories(saveFolder.str());
	}

	// Intermediate images (input, depth filter)
	//for (int idx = 0; idx < keyframeVector.size(); ++idx)
	//{
	//	auto& pose = keyframeVector[idx].second;

	//	cv::Mat depthFiltered = cv::Mat(imageSize, CV_8UC3);
	//	cv::Mat input = cv::Mat(imageSize, CV_8UC3);
	//	std::vector<cv::Mat> depths;

	//	int blocks = projector.projectCloudToImageOctreeCUDA(pose, input, depthFiltered, depths);

	//	std::stringstream ssDF;
	//	std::stringstream ssInput;
	//	ssDF << baseFolder << sceneId << "/" << calibration.getWidth() << "_" << downsampleRatio << "/ours/batch_0/df_" << idx << ".png";
	//	ssInput << baseFolder << sceneId << "/" << calibration.getWidth() << "_" << downsampleRatio << "/ours/batch_0/input_" << idx << ".png";

	//	cv::imwrite(ssDF.str(), depthFiltered);
	//	cv::imwrite(ssInput.str(), input);

	//	//int i = 0;
	//	//for (auto& depthImage : depths)
	//	//{
	//	//	std::cout << i << "/" << depths.size() << std::endl;
	//	//	std::stringstream ssDF;
	//	//	ssDF << baseFolder << sceneId << "/" << calibration.getWidth() << "_" << downsampleRatio << "/ours/batch_0/depth_" << idx << "_" << i++ << ".png";

	//	//	//cv::imshow("Img", depthImage);
	//	//	//cv::waitKey(0);
	//	//	cv::Mat mask = depthImage > 50.0f | depthImage < 0;
	//	//	double minVal;
	//	//	cv::minMaxLoc(depthImage, &minVal, nullptr, nullptr, nullptr, ~mask);

	//	//	std::cout << minVal << std::endl;
	//	//	depthImage.setTo(minVal, mask);

	//	//	cv::Mat depthNormalized;
	//	//	cv::normalize( depthImage, depthNormalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);

	//	//	cv::imwrite(ssDF.str(), depthNormalized);
	//	//}

	//}

	// run 5 times so JIT is ready
	for (int i = 0; i < 5; ++i)
	{
		auto& pose = keyframeVector[0].second;
		cv::Mat render = cv::Mat(imageSize, CV_16FC3);

        int blocks = projector.projectCloudToImageOctreeCUDA(pose, &render);
	}


	for(int idx = 0; idx < keyframeVector.size()-1; ++idx)
	{
		auto& pose0 = keyframeVector[idx].second;
		auto& pose1 = keyframeVector[idx+1].second;

		std::vector<cv::Matx44d> poses = interpolatePoses(pose0, pose1, 10);

		for (int j = 0; j < poses.size() - 1; ++j)
		{
			auto& pose = poses[j];

			auto start = std::chrono::high_resolution_clock::now();

			cv::Mat render = cv::Mat(imageSize, CV_16FC3);
            int blocks = projector.projectCloudToImageOctreeCUDA(pose, &render);

            auto end = std::chrono::high_resolution_clock::now();

			cv::Mat frame8U;
			render.convertTo(frame8U, CV_8UC3, 255.0);

			writer.write(frame8U);

			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			if (duration > maxDuration)
			{
				maxDuration = duration;
				maxBlocks = blocks;
			}

			sumDuration += duration;

			std::cout << idx << "\tExecution time: " << duration << "\tms\t"  << blocks << std::endl;

		// write images
		// std::stringstream ssRGB;
		// ssRGB << baseFolder << sceneId << "/" << calibration.getWidth() << "_" << downsampleRatio << "/ours/batch_0/rgb_" << idx << ".png";
		// cv::imwrite(ssRGB.str(), frame8U);
		}
	}

	std::cout << "\Max time: " << maxDuration << "\tms\t" << maxBlocks << "\tAvg:\t" << (float)1000.0/(float)(sumDuration/keyframeVector.size()) << std::endl;

	// Write data
	std::stringstream dataPath;
	dataPath << baseFolder << sceneId << "/" << calibration.getWidth() << "_" << downsampleRatio << "/ours/data.json";
	std::ofstream outFile(dataPath.str());
	if (outFile.is_open()) {
		outFile << "{" << "\"ids\" : [ ";
		for (int i = 0; i < keyframeVector.size(); i++)
		{
			outFile << keyframeVector[i].first;
			if (i < keyframeVector.size() - 1)
				outFile << ",";
		}
		outFile << "], \"total_time\" : " << (float)sumDuration / 1000.0 << ", \"avg_time\": " << (float)((float)sumDuration / 1000.0) / keyframeVector.size() << ", \"nrPoints\" : " << OctreeGrid::getNumberPoints(grid) << "}";
		outFile.close();
	}

#ifdef WIN32
    try {
        FreeLibrary(loadLib);
    }
    catch (...)
    {
        return 0;
    }
#endif
}
