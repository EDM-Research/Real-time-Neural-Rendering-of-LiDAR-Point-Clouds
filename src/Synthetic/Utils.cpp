#include "Utils.h"
#include <algorithm>
#include <fstream>
#include "RTRenderer/Config.h"


// Function to adjust brightness
cv::Vec3b adjustBrightness(const cv::Vec3b& pixel, int amount) {
	if (amount <= 0)
	{
		return pixel;
	}
	cv::Vec3b result;
	for (int i = 0; i < 3; ++i) {
		result[i] = cv::saturate_cast<uchar>(pixel[i] + amount);
	}
	return result;
}

// Function to adjust contrast
cv::Vec3b adjustContrast(const cv::Vec3b& pixel, float factor) {
	if (factor <= 0.0)
	{
		return pixel;
	}
	cv::Vec3b result;
	for (int i = 0; i < 3; ++i) {
		result[i] = cv::saturate_cast<uchar>(128 + factor * (pixel[i] - 128));
	}
	return result;
}

cv::Vec3b addGaussianNoise(const cv::Vec3b& pixel, float sigma) {
	cv::Vec3b result;
	for (int i = 0; i < 3; ++i) {
		int noise = static_cast<int>(pixel[i] + cv::theRNG().gaussian(sigma));
		result[i] = cv::saturate_cast<uchar>(noise);
	}
	return result;
}

std::vector<cv::Point> generateRandomConvexShape(int imageWidth, int imageHeight) {
	// Define a maximum radius for the shape (smaller radius results in smaller shapes)
	int maxRadius = std::min(imageWidth, imageHeight) / 5; // Adjust this to make shapes smaller or larger

	// Choose a random center for the shape
	int centerX = rand() % imageWidth;
	int centerY = rand() % imageHeight;

	int numPoints = 3 + rand() % 5; // Random number of points (3 to 7)
	std::vector<cv::Point> points;

	// Generate random points around the center within a small radius
	for (int i = 0; i < numPoints; ++i) {
		// Create a random angle and distance for the point
		float angle = static_cast<float>(rand()) / RAND_MAX * 2 * CV_PI; // Random angle [0, 2π]
		float distance = static_cast<float>(rand() % maxRadius) + 5;      // Random distance [5, maxRadius]

		// Compute point coordinates with a slight jitter effect
		int x = static_cast<int>(centerX + distance * cos(angle) + (rand() % 10 - 5)); // Jitter with ±5 pixels
		int y = static_cast<int>(centerY + distance * sin(angle) + (rand() % 10 - 5)); // Jitter with ±5 pixels

		// Ensure points are within the image boundaries
		x = std::clamp(x, 0, imageWidth - 1);
		y = std::clamp(y, 0, imageHeight - 1);

		points.emplace_back(x, y);
	}

	// Find the convex hull
	std::vector<cv::Point> convexHull;
	cv::convexHull(points, convexHull);

	return convexHull;
}


// Function to check if a point is inside a convex polygon
bool isInsidePolygon(int x, int y, const std::vector<cv::Point>& polygon) {
	return cv::pointPolygonTest(polygon, cv::Point(x, y), false) >= 0;
}

cv::Mat generateNoisyVariant(const cv::Mat& image) {
	cv::Mat noisyImage = image.clone();

	int numNoiseParameters = rand() % 4 + 2;
	std::vector<std::pair<int, float>> noiseParameters;
	for (int i = 0; i < numNoiseParameters; ++i) {
		int brightness = (rand() % 41) - 20;
		float contrast = 0.8f + static_cast<float>(rand() % 41) / 100.0f; 
		noiseParameters.push_back({ brightness, contrast });
	}

	int numShapes = rand() % 6;
	std::vector<std::vector<cv::Point>> convexShapes;

	for (int i = 0; i < numShapes; ++i) {
		std::vector<cv::Point> shape = generateRandomConvexShape(image.cols, image.rows);
		convexShapes.push_back(shape);
	}

	for (int y = 0; y < image.rows; ++y) {
		for (int x = 0; x < image.cols; ++x) {
			cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);

			if (pixel == cv::Vec3b(0, 0, 0))
				continue;

			bool insideShape = false;
			for (const auto& shape : convexShapes) {
				if (isInsidePolygon(x, y, shape)) {
					insideShape = true;
					break;
				}
			}

			if (!insideShape && rand() % 100 >= 50)
				continue;

			const auto& params = noiseParameters[rand() % numNoiseParameters];

			pixel = adjustBrightness(pixel, params.first);
			pixel = adjustContrast(pixel, params.second);
			if (rand() % 100 >= 20) {
				pixel = addGaussianNoise(pixel, 10.0f);
			}

			noisyImage.at<cv::Vec3b>(y, x) = pixel;
		}
	}

	return noisyImage;
}



cv::Scalar generateColor(const cv::Mat& image) {
	if (rand() % 2 == 0) {  // 50% chance for black
		return cv::Scalar(0, 0, 0);  // Black color
	}
	else {
		int x = rand() % image.cols;
		int y = rand() % image.rows;
		return image.at<cv::Vec3b>(y, x);
	}
}


// Function to generate random dense clusters of square-like noise
void addDenseNoiseClusters(cv::Mat& image, cv::Mat& depth, int numClusters, int clusterSize) {

	for (int i = 0; i < numClusters; ++i) {
		int xCenter = rand() % image.cols;
		int yCenter = rand() % image.rows;
		cv::Scalar color = generateColor(image);

		float depthDelta = rand() % 51 + 10;


		for (int j = 0; j < clusterSize; ++j) {
			int xOffset = (rand() % 5) - 2;
			int yOffset = (rand() % 5) - 2;
			int x = std::clamp(xCenter + xOffset, 0, image.cols - 1);
			int y = std::clamp(yCenter + yOffset, 0, image.rows - 1);
			int squareSize = rand() % 2 + 1;  // Random size from 1 to 3

			float variation = rand() % 8 + 3;      // Generates a random variation between 3 and 10


			for (int dx = 0; dx < squareSize; ++dx) {
				for (int dy = 0; dy < squareSize; ++dy) {
					int xSquare = std::clamp(x + dx, 0, image.cols - 1);
					int ySquare = std::clamp(y + dy, 0, image.rows - 1);
					image.at<cv::Vec3b>(ySquare, xSquare) = cv::Vec3b(color[0], color[1], color[2]);
					if (color == cv::Scalar(0, 0, 0))
					{
						depth.at<float>(ySquare, xSquare) = -1;
					}
					else
					{
						depth.at<float>(ySquare, xSquare) -= (depthDelta + (rand() % 2 == 0) ? variation : -variation) / 500.0;
					}
				}
			}
		}
	}
}

// Function to generate random sparse larger square-like clusters of noise
void addSparseNoiseClusters(cv::Mat& image, cv::Mat& depth, int numClusters, int clusterSize) {
	for (int i = 0; i < numClusters; ++i) {
		int xCenter = rand() % image.cols;
		int yCenter = rand() % image.rows;
		cv::Scalar color = generateColor(image);

		float depthDelta = rand() % 51 + 10;

		for (int j = 0; j < clusterSize; ++j) {
			int xOffset = (rand() % 35) - 7;
			int yOffset = (rand() % 35) - 7;
			int x = std::clamp(xCenter + xOffset, 0, image.cols - 1);
			int y = std::clamp(yCenter + yOffset, 0, image.rows - 1);
			int squareSize = rand() % 2 + 1;  // Random size from 1 to 4

			float variation = rand() % 8 + 3;      // Generates a random variation between 3 and 10

			for (int dx = 0; dx < squareSize; ++dx) {
				for (int dy = 0; dy < squareSize; ++dy) {
					int xSquare = std::clamp(x + dx, 0, image.cols - 1);
					int ySquare = std::clamp(y + dy, 0, image.rows - 1);
					image.at<cv::Vec3b>(ySquare, xSquare) = cv::Vec3b(color[0], color[1], color[2]);
					if (color == cv::Scalar(0, 0, 0))
					{
						depth.at<float>(ySquare, xSquare) = -1;
					}
					else {
						depth.at<float>(ySquare, xSquare) += (depthDelta + (rand() % 2 == 0) ? variation : -variation)/500.0;
					}
				}
			}
		}
	}
}


void Utils::processImage(const std::unordered_map<int, OctreeGrid::Block>& grid, const cv::Size& imageSize, const cv::Matx44d& cameraPose, const cv::Matx33d intrinsics, const std::string& inputStr, cv::Mat& inputImg, cv::Mat& outputImg, bool realRender)
{
    // // get the original image
    // cv::Mat origImg = cv::imread(inputStr);
    // cv::resize(origImg, outputImg, imageSize);

    // inputImg = cv::Mat::zeros(imageSize, CV_8UC3);

    // cv::Mat mask = cv::Mat::zeros(imageSize, CV_8U);
    // cv::Mat depth = cv::Mat(imageSize, CV_32F, std::numeric_limits<float>::max());

    // if (realRender)
    // {
    // 	std::cout << "Real\n";

    // 	cv::Mat render = cv::Mat::zeros(imageSize, CV_8UC3);
    // 	Utils::projectCloudToImageOctree(grid, cameraPose, intrinsics, render, depth, mask);

    // 	//cv::Mat filteredDepth = reduceAndFilter(depth);
    // 	cv::Mat outputMask = depth != std::numeric_limits<float>::max();

    // 	render.copyTo(inputImg, outputMask);

    // 	cv::Mat notOutputMask;
    // 	cv::bitwise_not(outputMask, notOutputMask);

    // 	double min, max;
    // 	cv::minMaxLoc(depth, &min, &max);

    // 	depth.setTo(min, notOutputMask);
    // 	cv::Mat tooFar = (depth == std::numeric_limits<float>::max());
    // 	depth.setTo(min, tooFar);
    // 	cv::normalize(depth, depth, 0, 1, cv::NORM_MINMAX);
    // 	depth.setTo(-1, tooFar);
    // 	depth.setTo(-1, notOutputMask);

    // 	mask = outputMask;
    // }
    // else
    // {

    // 	std::cout << "Synthetic\n";
    // 	cv::Mat color = cv::Mat::zeros(imageSize, CV_8UC3);

    // 	Utils::projectCloudToImageOctree(grid, cameraPose, intrinsics, color, depth, mask);

    // 	cv::Mat filteredDepth = reduceAndFilter(depth);
		
    // 	cv::Mat background = filteredDepth == std::numeric_limits<float>::max() & mask > 0;
		
    // 	cv::Mat outputMask = depth != std::numeric_limits<float>::max();

    // 	origImg.copyTo(inputImg, outputMask);
    // 	color.copyTo(inputImg, background);

    // 	cv::Mat notOutputMask;
    // 	cv::bitwise_not(outputMask, notOutputMask);

    // 	double min, max;
    // 	cv::minMaxLoc(depth, &min, &max);

    // 	depth.setTo(min, notOutputMask);
    // 	cv::Mat tooFar = (depth == std::numeric_limits<float>::max());
    // 	depth.setTo(min, tooFar);
    // 	cv::normalize(depth, depth, 0, 1, cv::NORM_MINMAX);
    // 	depth.setTo(-1, tooFar);
    // 	depth.setTo(-1, notOutputMask);

    // 	mask = outputMask;
    // }


    // if (!realRender)
    // {
    // 	inputImg = generateNoisyVariant(inputImg);
    // 	addDenseNoiseClusters(inputImg, depth, 30, 15);
    // 	addSparseNoiseClusters(inputImg, depth, 30, 15);
    // }

    // convertToFloat(depth, mask, inputImg);
}

void Utils::convertToFloat(cv::Mat& depth, cv::Mat& mask, cv::Mat& inputImg)
{
	mask.convertTo(mask, CV_32F);
	mask /= 255.0;

	inputImg.convertTo(inputImg, CV_32FC3);
	inputImg /= 255.0;

	std::vector<cv::Mat> channels;
	cv::split(inputImg, channels);
	std::reverse(channels.begin(), channels.end());
	channels.push_back(mask);
	channels.push_back(depth);
	cv::merge(channels, inputImg);
}

void Utils::writeToFile(const std::string& output_path, const cv::Mat& inputImg, const cv::Mat& outputImg, const std::string& split, const std::string& folder, const int frameId)
{
	std::stringstream ssOutput1;
	std::stringstream ssOutput2;

	ssOutput1 << output_path << "/" << split << "_input" << "/" << folder << "_frame_" << std::setw(6) << std::setfill('0') << frameId;
	ssOutput2 << output_path << "/" << split << "_output" << "/" << folder << "_frame_" << std::setw(6) << std::setfill('0') << frameId << ".png";

	cv::imwrite(ssOutput2.str(), outputImg);


	std::vector<cv::Mat> channels;
	cv::split(inputImg, channels);
	if (channels.size() % 4 <= 1)
	{
		for (int i = 0; i <= channels.size() / 4; ++i)
		{
			int numChannels = std::min(int(channels.size() - i * 4), 4);
			std::vector<cv::Mat> vecToSave(channels.begin() + i * 4, channels.begin() + i * 4 + numChannels);
			cv::Mat imgToSave;
			cv::merge(vecToSave, imgToSave);

			cv::imwrite(ssOutput1.str() + "_" + std::to_string(i) + ".exr", imgToSave);
		}
	}
	else if (channels.size() % 3 <= 1)
	{
		for (int i = 0; i <= channels.size() / 3; ++i)
		{
			int numChannels = std::min(int(channels.size() - i * 3), 3);
			std::vector<cv::Mat> vecToSave(channels.begin() + i * 3, channels.begin() + i * 3 + numChannels);
			cv::Mat imgToSave;
			cv::merge(vecToSave, imgToSave);
			cv::imwrite(ssOutput1.str() + "_" + std::to_string(i) + ".exr", imgToSave);
		}
	}
}

