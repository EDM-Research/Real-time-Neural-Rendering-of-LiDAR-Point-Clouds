#pragma once
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include "Octreegrid.h"
#include "CameraCalibration.h"
#include <torch/script.h>
#include <torch/cuda.h>
#include <glm/glm.hpp>
#include "types.h"

class ProjectCloud {
public:
    ProjectCloud(const std::unordered_map<int, OctreeGrid::Block>& grid, const std::string& modelFilename);
    ~ProjectCloud();

    int computeRGBD(const CameraCalibration& calibration, const cv::Matx44d& extrinsics, cv::Mat* color, cv::Mat* depth);
    int computeFilteredRGBD(const CameraCalibration& calibration, const cv::Matx44d& extrinsics, cv::Mat* color, cv::Mat* depth);
    int computeFull(const CameraCalibration& calibration, const cv::Matx44d& extrinsics, cv::Mat* color, cv::Mat* depth);

private:
    torch::jit::Module model;

    // Device buffers
    float4 *d_vertices_data;
    uchar4 *d_color_data;
    uint8_t *d_image;
    glm::mat4 *d_cam_proj;
    uint32_t *d_local_maxes;
    uint32_t *d_local_mins;
    uint32_t *d_final_max;
    uint32_t *d_final_min;
    c10::Half* tensorPtr;
    uint32_t* d_output_color;
    uint32_t* d_output_depth;

    size_t data_size;
    uint2 image_size;

    int min_grid_size;
    int block_size;
    int numBlocksPCDLevel;
    int numBlocksImgLevel;

    dim3 block_dim;
    dim3 grid_dim;

    int computeRGBDInternal(const CameraCalibration& calibration, const cv::Matx44d& extrinsics);
    void applyDepthFilter(const CameraCalibration& calibration);

    glm::mat4 cvMatx44dToGlmMat4(const cv::Matx44d& mat) {
        glm::mat4 result;
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                result[col][row] = static_cast<float>(mat(row, col));
                // glm is column-major: result[column][row]
            }
        }
        return result;
    }
};
