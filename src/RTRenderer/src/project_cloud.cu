#include "project_cloud.h"
#include <stdio.h>
#include "render.cuh"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/fwd.hpp>
#include <glm/matrix.hpp>
#include "Timer.h"
#include "types.h"
#include <cuda_runtime.h>
#include <filesystem>

#define CUDA_ERROR(x)                                                                      \
    if (x != cudaSuccess) {                                                                  \
    std::cerr << "CUDA ERROR (" << __LINE__ << "):" << cudaGetErrorString(x) << std::endl; \
    exit(1);                                                                              \
    }


#define THREADS_PER_BLOCK 1024
#define MAX_FLOAT 3.4028e38

const int depthRescaleDepth = 4;
__constant__ float filterStrength = 1.025;
__constant__ float gradientFilter = 0.03;
__constant__ int laplaceKernel[9] = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };

__global__ void reduce(float* highRes, float* lowRes, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < width * height)
    {
        int x = idx % width;
        int y = idx / width;

        int xHighRes = x * 2;
        int yHighRes = y * 2;
        int widthHighRes = width * 2;

        float pixel0 = highRes[yHighRes * widthHighRes + xHighRes];
        float pixel1 = highRes[yHighRes * widthHighRes + xHighRes + 1];
        float pixel2 = highRes[(yHighRes+1)* widthHighRes + xHighRes];
        float pixel3 = highRes[(yHighRes+1) * widthHighRes + xHighRes + 1];

        float lowest0 = pixel0 < pixel1 ? pixel0 : pixel1;
        float lowest1 = pixel2 < pixel3 ? pixel2 : pixel3;

        float lowest = lowest0 < lowest1 ? lowest0 : lowest1;

        lowRes[idx] = lowest;
    }
}

__global__ void laplacianKernel(const float* input, uint8_t* output, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < width * height)
    {
        int x = idx % width;
        int y = idx / width;

        if (x == 0 || x == width - 1 || y == 0 || y == height - 1)
        {
            output[idx] = 0;
            return;
        }

        float sum = 0;
        int counter = 0;
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                float fl = input[((y + ky) * width + (x + kx))];
                sum += fl * laplaceKernel[counter];
                counter++;
            }
        }
        output[idx] = (sum > gradientFilter) * 255;
    }
}

__device__ float getPixelValue(const float* lowRes, int x, int y, int width, int height)
{
    if (x >= 0 && x < width && y >= 0 && y < height)
        return lowRes[y * width + x];
    else return -1.0;
}

__global__ void compareImgsKernel(const float* lowRes, const float* highRes, const uint8_t* gradientMask, uint8_t* mask, int highResWidth, int highResHeight)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < highResHeight * highResWidth)
    {
        int highResX = idx % (highResWidth);
        int highResY = idx / (highResWidth);

        float currentVal = highRes[idx];
        if (currentVal >= MAX_FLOAT)
        {
            mask[idx] = 0;
            return;
        }

        int lowResX = highResX / 2;
        int lowResY = highResY / 2;

        int lowResWidth = highResWidth / 2;
        int lowResHeight = highResHeight / 2;

        if (gradientMask[lowResY * lowResWidth + lowResX] > 0)
        {
            if (currentVal <= getPixelValue(lowRes, lowResX - 1, lowResY - 1, lowResWidth, lowResHeight) * filterStrength || currentVal <= getPixelValue(lowRes, lowResX - 1, lowResY, lowResWidth, lowResHeight) * filterStrength || currentVal <= getPixelValue(lowRes, lowResX - 1, lowResY + 1, lowResWidth, lowResHeight) * filterStrength ||
                    currentVal <= getPixelValue(lowRes, lowResX, lowResY - 1, lowResWidth, lowResHeight) * filterStrength || currentVal <= getPixelValue(lowRes, lowResX, lowResY, lowResWidth, lowResHeight) * filterStrength || currentVal <= getPixelValue(lowRes, lowResX, lowResY + 1, lowResWidth, lowResHeight) * filterStrength ||
                    currentVal <= getPixelValue(lowRes, lowResX + 1, lowResY - 1, lowResWidth, lowResHeight) * filterStrength || currentVal <= getPixelValue(lowRes, lowResX + 1, lowResY, lowResWidth, lowResHeight) * filterStrength || currentVal <= getPixelValue(lowRes, lowResX + 1, lowResY + 1, lowResWidth, lowResHeight) * filterStrength)
            {
                mask[idx] = 255;
                return;
            }
        }
        else if(currentVal <= getPixelValue(lowRes, lowResX, lowResY, lowResWidth, lowResHeight) * filterStrength)
        {
            mask[idx] = 255;
            return;
        }
        mask[idx] = 0;// moet 0 zijn
    }
}

__global__ void resizeKernel(const float* lowRes, float* highRes, const uint8_t* mask, int outWidth, int outHeight) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= outWidth * outHeight) return;

    if (mask[idx] > 0) return;

    int x = idx % outWidth;
    int y = idx / outWidth;

    float inX = (x + 0.5f) / 2.0f - 0.5f;
    float inY = (y + 0.5f) / 2.0f - 0.5f;

    int lowWidth = outWidth / 2;
    int lowHeight = outHeight / 2;

    int x0 = (int)floorf(inX);
    int x1 = x0 + 1;
    int y0 = (int)floorf(inY);
    int y1 = y0 + 1;

    x0 = (x0 < 0) ? 0 : (x0 >= lowWidth ? lowWidth - 1 : x0);
    x1 = (x1 < 0) ? 0 : (x1 >= lowWidth ? lowWidth - 1 : x1);
    y0 = (y0 < 0) ? 0 : (y0 >= lowHeight ? lowHeight - 1 : y0);
    y1 = (y1 < 0) ? 0 : (y1 >= lowHeight ? lowHeight - 1 : y1);

    float wx = inX - x0;
    float wy = inY - y0;

    float v0 = (1 - wx) * lowRes[y0 * lowWidth + x0] + wx * lowRes[y0 * lowWidth + x1];
    float v1 = (1 - wx) * lowRes[y1 * lowWidth + x0] + wx * lowRes[y1 * lowWidth + x1];

    highRes[idx] = (1 - wy) * v0 + wy * v1;
}

__global__ void removeMask(float* depthBuffer, uint8_t* renderBuffer, uint8_t* mask, c10::Half* output, uint32_t* min, uint32_t* max, int width, int height)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < width * height)
    {
        if (mask[idx] == 0)
        {
            depthBuffer[idx] = (float)-1.f;
            renderBuffer[idx * 3] = 0;
            renderBuffer[idx * 3 + 1] = 0;
            renderBuffer[idx * 3 + 2] = 0;
            output[width * height * 0 + idx] = (c10::Half) 0.0f;
            output[width * height * 1 + idx] = (c10::Half) 0.0f;
            output[width * height * 2 + idx] = (c10::Half) 0.0f;
            output[width * height * 3 + idx] = (c10::Half)0.0f;
            output[width * height * 4 + idx] = (c10::Half)-1.f;
            return;
        }
        output[width * height * 0 + idx] = (c10::Half)renderBuffer[idx * 3] / 255.0f;
        output[width * height * 1 + idx] = (c10::Half)renderBuffer[idx * 3 + 1] / 255.0f;
        output[width * height * 2 + idx] = (c10::Half)renderBuffer[idx * 3 + 2] / 255.0f;
        output[width * height * 3 + idx] = (c10::Half)mask[idx] / 255.0f;
        output[width * height * 4 + idx] = (c10::Half)(depthBuffer[idx] - __uint_as_float(*min)) / (__uint_as_float(*max) - __uint_as_float(*min));
    }
}

ProjectCloud::ProjectCloud(const std::unordered_map<int, OctreeGrid::Block>& grid, const std::string& modelFilename)
{
    std::vector<float4> vertices = OctreeGrid::getVertexPositions(grid);
    std::vector<uchar4> colors = OctreeGrid::getVertexColors(grid);

    data_size = vertices.size();
    image_size = {1,1};
    block_dim.x = 16;
    block_dim.y = 16;
    block_dim.z = 1;

    CUDA_ERROR(cudaMalloc(&d_vertices_data, data_size * sizeof(float4)));
    CUDA_ERROR(cudaMalloc(&d_color_data, data_size * sizeof(uint4)));
    CUDA_ERROR(cudaMalloc(&d_cam_proj, sizeof(glm::mat4)));
    CUDA_ERROR(cudaMalloc(&d_final_max, sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_final_min, sizeof(uint32_t)));
    CUDA_ERROR(cudaMemcpy(d_vertices_data, vertices.data(), data_size * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_color_data, colors.data(), data_size * sizeof(uchar4), cudaMemcpyHostToDevice));

    CUDA_ERROR(cudaMalloc(&d_local_maxes, sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_local_mins, sizeof(uint32_t)));

    CUDA_ERROR(cudaMalloc(&d_output_color, sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_output_depth, sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_image, sizeof(uint8_t) * 3));
    CUDA_ERROR(cudaMalloc(&tensorPtr, sizeof(c10::Half)));

    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, minDepthPass);
    numBlocksPCDLevel = (data_size + block_size - 1) / block_size;

    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available!" << std::endl;
    }
    else {
        std::cout << "CUDA is NOT available!" << std::endl;
    }
    if(modelFilename != std::string(""))
    {
        if(std::filesystem::exists(modelFilename))
        {
            std::cout << "Loading model from file: " << modelFilename << std::endl;
        }
        else
        {
            std::cerr << "Model file does not exist: " << modelFilename << ", make sure to compile the model for this camera resolution using model/export_ts.py with TensorRT support or model/export_pt.py without TensorRT support. See README.md for details." << std::endl;
            exit(-1);
        }
        try {
            model = torch::jit::load(modelFilename);
            model.to(torch::kCUDA);
            // model.to(torch::kHalf);
        }
        catch (const c10::Error& e) {
            std::cerr << "Error loading the model\n" << e.what() << std::endl << std::endl;
            exit(-1);
        }
    }
    else
    {
        std::cerr << "No model file name given, computeFull will not work." << std::endl;
    }
}

ProjectCloud::~ProjectCloud()
{
    cudaFree(d_vertices_data);
    cudaFree(d_color_data);
    cudaFree(d_output_color);
    cudaFree(d_output_depth);
    cudaFree(d_image);
    cudaFree(d_cam_proj);
    cudaFree(d_local_maxes);
    cudaFree(d_local_mins);
    cudaFree(d_final_max);
    cudaFree(d_final_min);
    cudaFree(tensorPtr);
}

int ProjectCloud::computeRGBD(const CameraCalibration &calibration, const cv::Matx44d &extrinsics, cv::Mat *color, cv::Mat *depth)
{
    if(color == nullptr && depth == nullptr)
    {
        return -1;
    }

    if(calibration.getWidth() != image_size.x || calibration.getHeight() != image_size.y)
    {
        cudaFree(d_output_color);
        cudaFree(d_image);
        cudaFree(d_output_depth);
        cudaFree(d_local_maxes);
        cudaFree(d_local_mins);
        cudaFree(tensorPtr);

        numBlocksImgLevel = (calibration.getWidth() * calibration.getHeight() + block_size - 1) / block_size;
        CUDA_ERROR(cudaMalloc(&d_local_maxes, numBlocksImgLevel * sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc(&d_local_mins, numBlocksImgLevel * sizeof(uint32_t)));

        CUDA_ERROR(cudaMalloc(&d_output_color, calibration.getWidth() * calibration.getHeight() * sizeof(uint32_t) * 4));
        CUDA_ERROR(cudaMalloc(&d_image, calibration.getWidth() * calibration.getHeight() * sizeof(uint8_t) * 3));
        CUDA_ERROR(cudaMalloc(&d_output_depth, calibration.getWidth() * calibration.getHeight() * sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc(&tensorPtr, calibration.getWidth() * calibration.getHeight() * 5 * sizeof(c10::Half)));

        grid_dim.x = calibration.getWidth() / block_dim.x;
        grid_dim.y = calibration.getHeight() / block_dim.y;
        grid_dim.z = 1;

        image_size = {static_cast<unsigned int>(calibration.getWidth()), static_cast<unsigned int>(calibration.getHeight())};
    }

    computeRGBDInternal(calibration, extrinsics);

    if(depth != nullptr)
    {
        cudaMemcpy(depth->ptr<float>(), d_output_depth, calibration.getWidth() * calibration.getHeight() * sizeof(float), cudaMemcpyDeviceToHost);
    }
    if(color != nullptr)
    {
        cudaMemcpy(color->ptr<uint8_t>(), d_image, calibration.getWidth() * calibration.getHeight() * sizeof(uint8_t) * 3, cudaMemcpyDeviceToHost);
    }

    return 1;
}

int ProjectCloud::computeRGBDInternal(const CameraCalibration &calibration, const cv::Matx44d &extrinsics)
{
    fillBuffer<<<grid_dim, block_dim>>>(d_output_depth, 0x7F7FFFFF, calibration.getWidth() * calibration.getHeight());
    cudaMemset(d_output_color, 0, calibration.getWidth() * calibration.getHeight()*4 * sizeof(uint32_t));
    glm::mat4 camProj = glm::transpose(glm::mat4(glm::transpose(calibration.getGlmIntrinsicsMatrix())) * cvMatx44dToGlmMat4(extrinsics));

    CUDA_ERROR(cudaMemcpy(d_cam_proj, glm::value_ptr(camProj), sizeof(glm::mat4), cudaMemcpyHostToDevice));
    minDepthPass<<<numBlocksPCDLevel, block_size>>>(d_output_depth, d_vertices_data, (float*)d_cam_proj, image_size, data_size);
    cudaDeviceSynchronize();
    accumulatePass<<<numBlocksPCDLevel, block_size>>>(d_vertices_data, d_color_data, data_size, image_size, (float*)d_cam_proj, d_output_depth, d_output_color);
    cudaDeviceSynchronize();
    resolvePass<<<grid_dim, block_dim>>>(d_image, d_output_color, calibration.getWidth() * calibration.getHeight());
    cudaDeviceSynchronize();

    return 1;
}

void ProjectCloud::applyDepthFilter(const CameraCalibration &calibration)
{
    std::vector<float*> imgResolutions(depthRescaleDepth + 1);
    imgResolutions[0] = reinterpret_cast<float*>(d_output_depth);

    int newWidth = calibration.getWidth();
    int newHeight = calibration.getHeight();
    int nrBlocks = (int)((newWidth * newHeight + block_size - 1) / block_size);

    for (int i = 1; i < imgResolutions.size(); ++i)
    {
        newWidth /= 2;
        newHeight /= 2;
        nrBlocks = (int)((newWidth * newHeight + block_size - 1) / block_size);

        cudaMalloc(&imgResolutions[i], (newWidth * newHeight) * sizeof(float));

        reduce<<<nrBlocks, block_size>>>(imgResolutions[i - 1], imgResolutions[i], newWidth, newHeight);
        cudaDeviceSynchronize();
    }

    for (int i = imgResolutions.size() - 1; i >= 1; --i)
    {
        uint8_t* gradMask;
        cudaMalloc(&gradMask, (newWidth * newHeight) * sizeof(uint8_t));

        laplacianKernel<<<nrBlocks, block_size>>>(imgResolutions[i], gradMask, newWidth, newHeight);
        cudaDeviceSynchronize();

        newWidth *= 2;
        newHeight *= 2;
        nrBlocks = (int)((newWidth * newHeight + block_size - 1) / block_size);

        uint8_t* mask;
        cudaMalloc(&mask, newWidth * newHeight * sizeof(uint8_t));

        compareImgsKernel<<<nrBlocks, block_size>>>(imgResolutions[i], imgResolutions[i - 1], gradMask, mask, newWidth, newHeight);
        cudaDeviceSynchronize();

        cudaFree(gradMask);

        if (i == 1)
        {
            uint32_t sharedMemSize = 2 * block_size * sizeof(uint32_t);
            find_local_minmax_kernel<<<nrBlocks, block_size, sharedMemSize>>>(d_output_depth, d_local_mins, d_local_maxes, newWidth * newHeight);
            cudaDeviceSynchronize();
            find_overall_minmax_kernel<<<1, block_size, sharedMemSize>>>(d_local_mins, d_local_maxes, d_final_min, d_final_max, nrBlocks);
            cudaDeviceSynchronize();

            removeMask<<<nrBlocks, block_size>>>(imgResolutions[i - 1], d_image, mask, tensorPtr, d_final_min, d_final_max, newWidth, newHeight);
            cudaDeviceSynchronize();
        }
        else
        {
            resizeKernel<<<nrBlocks, block_size>>>(imgResolutions[i], imgResolutions[i - 1], mask, newWidth, newHeight);
            cudaDeviceSynchronize();
        }

        cudaFree(imgResolutions[i]);
        cudaFree(mask);
    }
}

int ProjectCloud::computeFilteredRGBD(const CameraCalibration& calibration, const cv::Matx44d& extrinsics, cv::Mat* color, cv::Mat* depth)
{
    if(calibration.getWidth() != image_size.x || calibration.getHeight() != image_size.y)
    {
        cudaFree(d_output_color);
        cudaFree(d_image);
        cudaFree(d_output_depth);
        cudaFree(d_local_maxes);
        cudaFree(d_local_mins);
        cudaFree(tensorPtr);

        numBlocksImgLevel = (calibration.getWidth() * calibration.getHeight() + block_size - 1) / block_size;
        CUDA_ERROR(cudaMalloc(&d_local_maxes, numBlocksImgLevel * sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc(&d_local_mins, numBlocksImgLevel * sizeof(uint32_t)));

        CUDA_ERROR(cudaMalloc(&d_output_color, calibration.getWidth() * calibration.getHeight() * sizeof(uint32_t) * 4));
        CUDA_ERROR(cudaMalloc(&d_image, calibration.getWidth() * calibration.getHeight() * sizeof(uint8_t) * 3));
        CUDA_ERROR(cudaMalloc(&d_output_depth, calibration.getWidth() * calibration.getHeight() * sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc(&tensorPtr, calibration.getWidth() * calibration.getHeight() * 5 * sizeof(c10::Half)));

        grid_dim.x = calibration.getWidth() / block_dim.x;
        grid_dim.y = calibration.getHeight() / block_dim.y;
        grid_dim.z = 1;

        image_size = {static_cast<unsigned int>(calibration.getWidth()), static_cast<unsigned int>(calibration.getHeight())};
    }

    computeRGBDInternal(calibration, extrinsics);
    applyDepthFilter(calibration);

    if(depth != nullptr)
    {
        cudaMemcpy(depth->ptr<float>(), d_output_depth, calibration.getWidth() * calibration.getHeight() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    if(color != nullptr)
    {
        cudaMemcpy(color->ptr<uint8_t>(), d_image, calibration.getWidth() * calibration.getHeight() * sizeof(uint8_t) * 3, cudaMemcpyDeviceToHost);
    }

    return 1;
}


int ProjectCloud::computeFull(const CameraCalibration& calibration, const cv::Matx44d& extrinsics, cv::Mat* color, cv::Mat* depth)
{
    auto start = std::chrono::high_resolution_clock::now();
    if(calibration.getWidth() != image_size.x || calibration.getHeight() != image_size.y)
    {
        cudaFree(d_output_color);
        cudaFree(d_image);
        cudaFree(d_output_depth);
        cudaFree(d_local_maxes);
        cudaFree(d_local_mins);
        cudaFree(tensorPtr);

        numBlocksImgLevel = (calibration.getWidth() * calibration.getHeight() + block_size - 1) / block_size;
        CUDA_ERROR(cudaMalloc(&d_local_maxes, numBlocksImgLevel * sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc(&d_local_mins, numBlocksImgLevel * sizeof(uint32_t)));

        CUDA_ERROR(cudaMalloc(&d_output_color, calibration.getWidth() * calibration.getHeight() * sizeof(uint32_t) * 4));
        CUDA_ERROR(cudaMalloc(&d_image, calibration.getWidth() * calibration.getHeight() * sizeof(uint8_t) * 3));
        CUDA_ERROR(cudaMalloc(&d_output_depth, calibration.getWidth() * calibration.getHeight() * sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc(&tensorPtr, calibration.getWidth() * calibration.getHeight() * 5 * sizeof(c10::Half)));

        grid_dim.x = calibration.getWidth() / block_dim.x;
        grid_dim.y = calibration.getHeight() / block_dim.y;
        grid_dim.z = 1;

        image_size = {static_cast<unsigned int>(calibration.getWidth()), static_cast<unsigned int>(calibration.getHeight())};
    }

    computeRGBDInternal(calibration, extrinsics);
    auto start_1 = std::chrono::high_resolution_clock::now();

    applyDepthFilter(calibration);
    auto start_2 = std::chrono::high_resolution_clock::now();

    torch::Tensor input = torch::from_blob(tensorPtr, {1,5, calibration.getHeight(),calibration.getWidth()}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));

    torch::NoGradGuard no_grad;
    auto output = model.forward({input}).toTensor();
    output = output[0].permute({1, 2, 0}).contiguous();
    if(color != nullptr)
    {
        cv::Mat renderF = cv::Mat(cv::Size(calibration.getWidth(), calibration.getHeight()), CV_16FC3);
        cudaMemcpy(renderF.ptr<at::Half>(), output.data_ptr(), calibration.getWidth() * calibration.getHeight() * sizeof(at::Half) * 3, cudaMemcpyDeviceToHost);
        renderF.convertTo(*color, CV_8UC3, 255.0);
    }

    if(depth != nullptr)
    {
        cudaMemcpy(depth->ptr<float>(), d_output_depth, calibration.getWidth() * calibration.getHeight() * sizeof(float), cudaMemcpyDeviceToHost);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "RENDER_TIME: projection[" << std::chrono::duration_cast<std::chrono::milliseconds>(start_1 - start).count() << "], filter[" << std::chrono::duration_cast<std::chrono::milliseconds>(start_2 - start_1).count() << "], unet[" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start_2).count() << "], Total[" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "]" << std::endl;

    return 1;
}
