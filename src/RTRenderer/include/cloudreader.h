#ifndef CLOUDREADER_H
#define CLOUDREADER_H

#include "Octreegrid.h"
#include <filesystem>
#include "CameraCalibration.h"

namespace CloudReader
{

std::unordered_map<int, OctreeGrid::Block> loadCloud(const std::filesystem::path& file_name, const std::filesystem::path& cache_dir = std::filesystem::path());
void loadCubemaps(const std::filesystem::path& file_name, std::vector<cv::Mat>& imgs, std::vector<cv::Mat>& depths, std::vector<cv::Matx44d>& w2cam, std::vector<CameraCalibration>& calibs);

}

#endif // CLOUDREADER_H
