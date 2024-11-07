#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <numeric>
#include <random>
#include <opencv2/opencv.hpp>

class OctreeGrid {

public:
	OctreeGrid();

	struct Block {
		std::vector<cv::Point3f> positions;
		std::vector<cv::Vec3b> colors;
		cv::Point3f bbMin;
		cv::Point3f bbMax;
	};

	static bool readBinary(const std::string& filename, std::vector<cv::Point3f>& cloud, std::vector<cv::Vec3b>& colors)
	{
		std::ifstream fin(filename, std::ios::in | std::ios::binary);
		if (!fin.is_open())
		{
			return false;
		}

		size_t numPoints;

		fin.read((char*)&numPoints, sizeof(size_t));

		cloud.resize(numPoints);
		colors.resize(numPoints);

		fin.read((char*)&cloud[0], numPoints * 3 * sizeof(float));
		fin.read((char*)&colors[0], numPoints * 3 * sizeof(uchar));


		fin.close();

		return true;
	}


	static int encodeKey(int x, int y, int z, int numBlocks_x, int numBlocks_y, int numBlocks_z) {
		return x + y * numBlocks_x + z * numBlocks_x * numBlocks_y;
	}


	static bool writeOctreeBinary(const std::string& filename, const std::unordered_map<int, OctreeGrid::Block>& grid, int numBlocks_x, int numBlocks_y, int numBlocks_z) {
		// writing octree to file
		std::ofstream fout(filename, std::ios::out | std::ios::binary);
		if (!fout.is_open())
		{
			return false;
		}

		int numBlocks = grid.size();
		//const size_t numPoints = cloud.size();
		fout.write((char*)&numBlocks_x, sizeof(int));
		fout.write((char*)&numBlocks_y, sizeof(int));
		fout.write((char*)&numBlocks_z, sizeof(int));
		fout.write((char*)&numBlocks, sizeof(int));
		for (auto& block_pair : grid) {
			fout.write((char*)&block_pair.first, sizeof(int)); // write key
			const size_t numPointsInBlock = block_pair.second.positions.size();
			fout.write((char*)&numPointsInBlock, sizeof(size_t)); //// write number of points in block
			fout.write((char*)block_pair.second.positions.data(), numPointsInBlock * 3 * sizeof(float)); // write points
			fout.write((char*)block_pair.second.colors.data(), numPointsInBlock * 3 * sizeof(uchar)); // write colors
			fout.write((char*)&block_pair.second.bbMin, sizeof(cv::Point3f)); //// write bbMin
			fout.write((char*)&block_pair.second.bbMax, sizeof(cv::Point3f)); //// write bbMax
		}
		fout.close();

		return true;
	}


	static bool readOctreeBinary(const std::string& filename, std::unordered_map<int, OctreeGrid::Block>& grid, int& numBlocks_x, int& numBlocks_y, int& numBlocks_z)
	{
		std::ifstream fin(filename, std::ios::in | std::ios::binary);
		if (!fin.is_open())
		{
			return false;
		}


		int numBlocks;
		fin.read((char*)&numBlocks_x, sizeof(int));
		fin.read((char*)&numBlocks_y, sizeof(int));
		fin.read((char*)&numBlocks_z, sizeof(int));
		fin.read((char*)&numBlocks, sizeof(int));

		for (int i = 0; i < numBlocks; ++i) {
			int key;
			size_t numPointsInBlock;
			fin.read((char*)&key, sizeof(int)); // write key
			fin.read((char*)&numPointsInBlock, sizeof(size_t)); // read numPointsInBlock

			grid[key].positions.resize(numPointsInBlock);
			grid[key].colors.resize(numPointsInBlock);
			fin.read((char*)&grid[key].positions[0], numPointsInBlock * 3 * sizeof(float));
			fin.read((char*)&grid[key].colors[0], numPointsInBlock * 3 * sizeof(uchar));
			fin.read((char*)&grid[key].bbMin, sizeof(cv::Point3f)); //// read bbMin
			fin.read((char*)&grid[key].bbMax, sizeof(cv::Point3f)); //// read bbMax
		}

		fin.close();

		return true;
	}

	static void decodeKey(int key, int& x, int& y, int& z, int numBlocks_x, int numBlocks_y, int numBlocks_z) {
		z = key / (numBlocks_x * numBlocks_y);
		key -= (z * numBlocks_x * numBlocks_y);
		y = key / numBlocks_x;
		x = key % numBlocks_x;
	}


	static void downSample(std::unordered_map<int, Block>& grid, double keep_percentage)
	{
		for (auto& pair : grid)
		{
			size_t size = pair.second.positions.size();
			size_t remove_count = static_cast<size_t>(size * (1 - keep_percentage));

			std::vector<size_t> indices(size);
			std::iota(indices.begin(), indices.end(), 0);

			std::random_device rd;
			std::mt19937 gen(rd());
			std::shuffle(indices.begin(), indices.end(), gen);

			std::vector<cv::Point3f> shuffled_positions(size);
			std::vector<cv::Vec3b> shuffled_colors(size);
			for (size_t i = 0; i < size; i++)
			{
				shuffled_positions[i] = pair.second.positions[indices[i]];
				shuffled_colors[i] = pair.second.colors[indices[i]];
			}

			pair.second.positions = std::vector<cv::Point3f>(shuffled_positions.begin(), shuffled_positions.end() - remove_count);
			pair.second.colors = std::vector<cv::Vec3b>(shuffled_colors.begin(), shuffled_colors.end() - remove_count);
		}
	}

	static int getNumberPoints(const std::unordered_map<int, Block>& grid)
	{
		size_t total_size = 0;
		for (auto& pair : grid)
		{
			size_t size = pair.second.positions.size();
			total_size += size;
		}
		return total_size;
	}
};
