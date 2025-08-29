#ifndef CLOUDREADER_H
#define CLOUDREADER_H

#include "Octreegrid.h"
#include <filesystem>

namespace CloudReader {

std::unordered_map<int, OctreeGrid::Block> loadCloud(const std::filesystem::path& file_name, const std::filesystem::path& cache_dir = std::filesystem::path());
}

#endif // CLOUDREADER_H
