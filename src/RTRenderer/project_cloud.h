#pragma once
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include "Octreegrid.h"
#include "CameraCalibration.h"
#include <torch/script.h>
#include <torch/cuda.h>

class ProjectCloud {
public:
    ProjectCloud(const std::unordered_map<int, OctreeGrid::Block>& grid, const std::string& modelFilename);
    ~ProjectCloud();

    int computeRGBD(const CameraCalibration& calibration, const cv::Matx44d& extrinsics, cv::Mat* color, cv::Mat* depth);
    int computeFilteredRGBD(const CameraCalibration& calibration, const cv::Matx44d& extrinsics, cv::Mat* color, cv::Mat* depth);
    int computeFull(const CameraCalibration& calibration, const cv::Matx44d& extrinsics, cv::Mat* color, cv::Mat* depth);
//    int depthFilterComparison(cv::Matx44d extrinsics, cv::Mat* color, cv::Mat* depth);

private:
    torch::jit::Module model;

    float* pointCloudPosCUDA;
    uint8_t* pointCloudColCUDA;
    int* blockOffsetsCUDA;
    int* blockSizesCUDA;
    int threadsPerBlock = 1024;
    float epsilon = 0.02f;

    std::unordered_map<int, OctreeGrid::Block> grid;

    std::unordered_map<int, std::pair<int, int>> blockIndicesMapping;

    int computeRGBDInternal(const CameraCalibration& calibration, const cv::Matx44d& extrinsics, uint8_t** color, float** depth);
    void applyDepthFilter(const CameraCalibration& calibration, uint8_t* color, float* depth, float** tensorPtr);
    int cull(const CameraCalibration& calibration, const cv::Matx44d& extrinsics, std::vector<int>& blockIds, std::vector<int>& blockO);

};
