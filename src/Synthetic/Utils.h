#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>

#include "RTRenderer/Octreegrid.h"

namespace Utils
{
    void processImage(const std::unordered_map<int, OctreeGrid::Block>& grid, const cv::Size& imageSize, const cv::Matx44d& cameraPose, const cv::Matx33d intrinsics, const std::string& inputStr, cv::Mat& inputImg, cv::Mat& outputImg, bool realRender);
    void convertToFloat(cv::Mat& depth, cv::Mat& mask, cv::Mat& inputImg);

    void writeToFile(const std::string& output_path, const cv::Mat& inputImg, const cv::Mat& outputImg, const std::string& split, const std::string& folder, const int frameId);
};

#endif // UTILS_H
