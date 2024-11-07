#ifndef CLOUDREADER_H
#define CLOUDREADER_H

#include "Octreegrid.h"
#include <filesystem>

namespace CloudReader {

std::string cache_dir = ".pcl_cache";

std::unordered_map<int, OctreeGrid::Block> loadCloud(const std::filesystem::path& file_name);

}

#endif // CLOUDREADER_H
