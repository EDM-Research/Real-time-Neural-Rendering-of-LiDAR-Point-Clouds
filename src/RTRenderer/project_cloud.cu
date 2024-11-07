#include "project_cloud.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include "Frustum.h"
#include <stdio.h>
#ifdef WITH_TENSOR_RT
#include <dlfcn.h>
#endif

#define THREADS_PER_BLOCK 1024
#define MAX_FLOAT 3.4028e38

const int depthRescaleDepth = 4;
__constant__ float filterStrength = 1.025;
__constant__ float gradientFilter = 0.03;
__constant__ int laplaceKernel[9] = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };

__device__ void calcMinMax(float* bufferMin, float* bufferMax, int size, float* minResult, float* maxResult)
{
	int id = threadIdx.x;
	int next_offset = 1;

	while (next_offset < size)
	{
		float min = bufferMin[id] < bufferMin[(id + next_offset) % size] ? bufferMin[id] : bufferMin[(id + next_offset) % size];

		float val0 = (bufferMax[id] >= MAX_FLOAT ? 0 : bufferMax[id]);
		float val1 = (bufferMax[(id + next_offset) % size] >= MAX_FLOAT) ? 0 : bufferMax[(id + next_offset) % size];
		float max = val0 > val1 ? val0 : val1;

		__syncthreads();
		bufferMin[id] = min;
		bufferMax[id] = max;
		__syncthreads();
		next_offset *= 2;
	}
	*minResult = bufferMin[id];
	*maxResult = bufferMax[id];
}

__device__ float atomicMinFloat(float* address, float val) {
	int* address_as_int = (int*)address;
	int old = *address_as_int, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_int, assumed,
			__float_as_int(fminf(val, __int_as_float(assumed))));
	} while (assumed != old);

	return __int_as_float(old);
}

__device__ float atomicMaxFloat(float* address, float val) {
	int* address_as_int = (int*)address;
	int old = *address_as_int, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_int, assumed,
			__float_as_int(fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);

	return __int_as_float(old);
}

__global__ void findMinMax(float* a, float size, float* resultMin, float* resultMax)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float bufferMin[THREADS_PER_BLOCK];
	__shared__ float bufferMax[THREADS_PER_BLOCK];
	if (id < size)
	{
		bufferMin[threadIdx.x] = a[id];
		bufferMax[threadIdx.x] = a[id];
	}
	else
	{
		bufferMin[threadIdx.x] = MAX_FLOAT;
		bufferMax[threadIdx.x] = MAX_FLOAT;
	}
	__syncthreads();

	float min, max;
	calcMinMax(bufferMin, bufferMax, blockDim.x, &min, &max);;

	if (threadIdx.x == 0)
	{
		atomicMinFloat(resultMin, min);
		atomicMaxFloat(resultMax, max);
	}
}

__global__ void initializeFloatBuffer(float* buffer, size_t size, float value) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		buffer[idx] = value;
	}
}

__global__ void render32toRender8(uint8_t* render8, uint32_t* render32, uint32_t* counter, size_t size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int counterIdx = idx / 3;
	if (idx < size * 3) {
		if (counter[counterIdx] == 0)
		{
			render8[idx] = 0;
		}
		else
		{
			render8[idx] = render32[idx] / counter[counterIdx];
		}
	}
}

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

__global__ void removeMask(float* depthBuffer, uint8_t* renderBuffer, uint8_t* mask, float* output, float min, float max, int width, int height)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < width * height)
	{
		if (mask[idx] == 0)
		{
			depthBuffer[idx] = -1;
			renderBuffer[idx * 3] = 0;
			renderBuffer[idx * 3 + 1] = 0;
			renderBuffer[idx * 3 + 2] = 0;
			output[width * height * 0 + idx] = (float) 0.0f;
			output[width * height * 1 + idx] = (float) 0.0f;
			output[width * height * 2 + idx] = (float) 0.0f;
			output[width * height * 3 + idx] = (float)0.0f;
			output[width * height * 4 + idx] = -1;
			return;
		}
		output[width * height * 0 + idx] = (float)renderBuffer[idx * 3] / 255.0;
		output[width * height * 1 + idx] = (float)renderBuffer[idx * 3 + 1] / 255.0;
		output[width * height * 2 + idx] = (float)renderBuffer[idx * 3 + 2] / 255.0;
		output[width * height * 3 + idx] = (float)mask[idx] / 255.0;
		output[width * height * 4 + idx] = (depthBuffer[idx] - min) / (max - min);
	}
}

__global__ void removeMaskNonDF(float* depthBuffer, uint8_t* renderBuffer, uint32_t* mask, float* output, float min, float max, int width, int height)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < width * height)
	{
		if (mask[idx] == 0)
		{
			depthBuffer[idx] = -1;
			renderBuffer[idx * 3] = 0;
			renderBuffer[idx * 3 + 1] = 0;
			renderBuffer[idx * 3 + 2] = 0;
			output[width * height * 0 + idx] = (float)0.0f;
			output[width * height * 1 + idx] = (float)0.0f;
			output[width * height * 2 + idx] = (float)0.0f;
			output[width * height * 3 + idx] = (float)0.0f;
			output[width * height * 4 + idx] = -1;
			return;
		}
		output[width * height * 0 + idx] = (float)renderBuffer[idx * 3] / 255.0;
		output[width * height * 1 + idx] = (float)renderBuffer[idx * 3 + 1] / 255.0;
		output[width * height * 2 + idx] = (float)renderBuffer[idx * 3 + 2] / 255.0;
		output[width * height * 3 + idx] = (float)1.0f;
		output[width * height * 4 + idx] = (depthBuffer[idx] - min) / (max - min);
	}
}

__global__ void fillLocks(int width, int height, int* locks, uint32_t* counterBuffer)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < width * height)
    {
        if(counterBuffer[idx] > 0)
        {
            locks[idx] = 2;
        }
    }
}

__device__ void acquireWriteLock(int *lock) {
    while (atomicCAS(lock, 0, -1) != 0) {}
}

__device__ void releaseWriteLock(int *lock) {
    __threadfence();
    atomicExch(lock, 0);
}

__global__ void renderBlocks(const float* pointCloudPos, const uint8_t* pointCloudCol, const int* blockOffsets,
						const int* blockSizes, const int* blockIds, const int* blockCountNumber, const int numBlocks,
                        const float* projectionMatrix, const int width, const int height, float epsilon, int* locks,
						float* depthBuffer, uint32_t* renderBuffer, uint32_t* counterBuffer)
{
    if (blockIdx.x >= numBlocks)
        return;

    int blockId = blockIds[blockIdx.x];
    int offsetInBlock = blockDim.x * blockCountNumber[blockIdx.x] + threadIdx.x;
	if (offsetInBlock >= blockSizes[blockId])
		return;

	int pointIdx = blockOffsets[blockId] + offsetInBlock;

	float x = pointCloudPos[pointIdx * 3];
	float y = pointCloudPos[pointIdx * 3 + 1];
	float z = pointCloudPos[pointIdx * 3 + 2];

	float px = projectionMatrix[0] * x + projectionMatrix[1] * y + projectionMatrix[2] * z + projectionMatrix[3];
	float py = projectionMatrix[4] * x + projectionMatrix[5] * y + projectionMatrix[6] * z + projectionMatrix[7];
	float pz = projectionMatrix[8] * x + projectionMatrix[9] * y + projectionMatrix[10] * z + projectionMatrix[11];

	if (pz <= 0.01) return;

	px /= pz;
	py /= pz;

	int u = static_cast<int>(px + 0.5);
	int v = static_cast<int>(py + 0.5);

	if (u >= 0 && u < width && v >= 0 && v < height)
	{
		int pixelIdx = v * width + u;

        if(locks[pixelIdx] == 2)
        {
            return;
        }

		uint8_t b = pointCloudCol[pointIdx * 3];
		uint8_t g = pointCloudCol[pointIdx * 3 + 1];
		uint8_t r = pointCloudCol[pointIdx * 3 + 2];

        while (atomicCAS(&locks[pixelIdx], 0, 1) != 0); // acquire lock
        float currDepth = depthBuffer[pixelIdx];

        if ((pz < currDepth + epsilon) && (pz > currDepth - epsilon))
        {
            depthBuffer[pixelIdx] = pz;
            counterBuffer[pixelIdx] += 1;
            renderBuffer[pixelIdx * 3] += b;
            renderBuffer[pixelIdx * 3 + 1] += g;
            renderBuffer[pixelIdx * 3 + 2] += r;
        }
        else if (pz <= currDepth - epsilon)
        {
            depthBuffer[pixelIdx] = pz;
            counterBuffer[pixelIdx] = 1;
            renderBuffer[pixelIdx * 3] = b;
            renderBuffer[pixelIdx * 3 + 1] = g;
            renderBuffer[pixelIdx * 3 + 2] = r;
        }

        __threadfence();
        atomicExch(&locks[pixelIdx], 0); // release lock
	}
}


__device__ float3 normalize(float3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x / len, v.y / len, v.z / len);
}

__device__ float3 deprojectPixel(int x, int y, float depth, float fx, float fy, float cx, float cy)
{
    float camX = (x - cx)/fx;
    float camY = (y - cy)/fy;

    return make_float3(camX * depth, camY * depth, depth);
}

__global__ void otherDepthFilter(float* depthBuffer, uint8_t* mask, float fx, float fy, float cx, float cy, int width, int height)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= width * height) return;

    int x = idx % width;
    int y = idx / width;

    int lowX = (x-3) < 0 ? 0 : x - 3;
    int lowY = (y-3) < 0 ? 0 : y - 3;
    int highX = (x+3) >= width ? width-1 : x + 3;
    int highY = (y+3) >= height ? height-1 : y + 3;

    float3 centerPoint = deprojectPixel(x, y, depthBuffer[idx], fx, fy, cx, cy);
    float3 vecPtoC = normalize(make_float3(-centerPoint.x, -centerPoint.y, -centerPoint.z));

    float solidAngle = 0.0f;

    for(int i = 0; i < 8; ++i)
    {
        int curr_x = x;
        int curr_y = y;
        int j = 0;

        int dx = (i < 4) ? -1 : 1;
        int dy = (i % 4 < 2) ? -1 : 1;

        bool primaryX = (i % 2 == 0);

        float maxDot = 0.0f;

        while ((primaryX && curr_x != lowX && curr_x != highX) ||
               (!primaryX && curr_y != lowY && curr_y != highY))
        {
            ++j;
            if (primaryX) {
                if (curr_x + dx < lowX || curr_x + dx > highX) break;
                curr_x += dx;
            } else {
                if (curr_y + dy < lowY || curr_y + dy > highY) break;
                curr_y += dy;
            }

            while ((primaryX && curr_y != lowY && curr_y != highY && abs(y - curr_y) < j) ||
                   (!primaryX && curr_x != lowX && curr_x != highX && abs(x - curr_x) < j))
            {
                if (primaryX) {
                    if (curr_y + dy < lowY || curr_y + dy > highY) break;
                    curr_y += dy;
                } else {
                    if (curr_x + dx < lowX || curr_x + dx > highX) break;
                    curr_x += dx;
                }
                float3 pixel = deprojectPixel(curr_x, curr_y, depthBuffer[curr_y * width + curr_x], fx, fy, cx, cy);
                float3 vec = normalize(make_float3(pixel.x - centerPoint.x, pixel.y - centerPoint.y, pixel.z - centerPoint.z));

                float dot = vec.x * vecPtoC.x + vec.y * vecPtoC.y + vec.z * vecPtoC.z;

                if(dot > maxDot)
                {
                    maxDot = dot;
                }
            }
        }
        solidAngle += acosf(maxDot);
    }

    if(solidAngle > 5.0f)
    {
        mask[idx] = 255;
    }
    else
    {
        mask[idx] = 0;
    }
}

ProjectCloud::ProjectCloud(const std::unordered_map<int, OctreeGrid::Block>& grid, const std::string& modelFilename) : grid{ grid }
{
	int numPoints = 0;
	int numBlocks = 0;
	for (const auto& block_pair : grid)
	{
		numBlocks++;
		numPoints += block_pair.second.positions.size();
	}

	cudaMalloc(&pointCloudPosCUDA, numPoints * sizeof(float) * 3);
	cudaMalloc(&pointCloudColCUDA, numPoints * sizeof(uint8_t) * 3);
	cudaMalloc(&blockOffsetsCUDA, numBlocks * sizeof(int));
	cudaMalloc(&blockSizesCUDA, numBlocks * sizeof(int));

	int* blockOffsets = new int[numBlocks];
	int* blockSizes = new int[numBlocks];

	int blockId = 0;
	for (const auto& block_pair : grid)
	{
		// load point cloud positions
		float* posArray = new float[block_pair.second.positions.size() * 3];
		for (int i = 0; i < block_pair.second.positions.size(); ++i)
		{
			posArray[i * 3] = block_pair.second.positions[i].x;
			posArray[i * 3 + 1] = block_pair.second.positions[i].y;
			posArray[i * 3 + 2] = block_pair.second.positions[i].z;
		}

		blockIndicesMapping.insert(std::make_pair(block_pair.first, std::make_pair(blockId, (int)((block_pair.second.positions.size() + threadsPerBlock - 1) / threadsPerBlock))));

		if (blockId == 0) blockOffsets[blockId] = 0;
		else blockOffsets[blockId] = blockOffsets[blockId - 1] + blockSizes[blockId - 1];

		cudaMemcpy(pointCloudPosCUDA + blockOffsets[blockId] * 3, posArray, block_pair.second.positions.size() * 3 * sizeof(float), cudaMemcpyHostToDevice);
		blockSizes[blockId] = block_pair.second.positions.size();

		delete[] posArray;

		uint8_t* colArray = new uint8_t[block_pair.second.colors.size() * 3];
		for (int i = 0; i < block_pair.second.positions.size(); ++i)
		{
			colArray[i * 3] = block_pair.second.colors[i][0];
			colArray[i * 3 + 1] = block_pair.second.colors[i][1];
			colArray[i * 3 + 2] = block_pair.second.colors[i][2];
		}
		cudaMemcpy(pointCloudColCUDA + blockOffsets[blockId] * 3, colArray, block_pair.second.colors.size() * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);

		delete[] colArray;

		++blockId;
	}

	cudaMemcpy(blockOffsetsCUDA, blockOffsets, numBlocks * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(blockSizesCUDA, blockSizes, numBlocks * sizeof(int), cudaMemcpyHostToDevice);

	delete[] blockOffsets;
	delete[] blockSizes;

	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available!" << std::endl;

#ifdef WITH_TENSOR_RT
        std::cout << "Loading libtorchtrt.so" << std::endl;
        void* handle = dlopen("libtorchtrt.so", RTLD_LAZY | RTLD_GLOBAL);
        if (!handle) {
            std::cerr << "Failed to load Torch-TensorRT: " << dlerror() << std::endl;
        }
#endif
	}
	else {
		std::cout << "CUDA is NOT available!" << std::endl;
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

ProjectCloud::~ProjectCloud()
{
	cudaFree(pointCloudPosCUDA);
	cudaFree(blockOffsetsCUDA);
	cudaFree(blockSizesCUDA);
    cudaFree(pointCloudColCUDA);
}

int ProjectCloud::computeRGBD(const CameraCalibration &calibration, const cv::Matx44d &extrinsics, cv::Mat *color, cv::Mat *depth)
{
    if(color == nullptr && depth == nullptr)
    {
        return -1;
    }

    uint8_t* colorBufferPtr; float* depthBufferPtr;
    int splatBlocks = computeRGBDInternal(calibration, extrinsics, &colorBufferPtr, &depthBufferPtr);

    if(depth != nullptr)
    {
        cudaMemcpy(depth->ptr<float>(), depthBufferPtr, calibration.getWidth() * calibration.getHeight() * sizeof(float), cudaMemcpyDeviceToHost);
    }
    if(color != nullptr)
    {
        cudaMemcpy(color->ptr<uint8_t>(), colorBufferPtr, calibration.getWidth() * calibration.getHeight() * sizeof(uint8_t) * 3, cudaMemcpyDeviceToHost);
    }

    cudaFree(colorBufferPtr);
    cudaFree(depthBufferPtr);

    return splatBlocks;
}

int ProjectCloud::computeRGBDInternal(const CameraCalibration &calibration, const cv::Matx44d &extrinsics, uint8_t**color, float**depth)
{
    std::vector<int> blockIds; // used in cuda kernel for each blockidx
    std::vector<int> blockO; // used in cuda kernel for each blockidx
    int splatBlocks = cull(calibration, extrinsics, blockIds, blockO);

    cv::Matx33d K = calibration.getIntrinsicsMatrix();
    cv::Matx34f mvp(cv::Mat(cv::Mat(K) * cv::Mat(extrinsics)(cv::Rect(0, 0, 4, 3))));

    int width = calibration.getWidth();
    int height = calibration.getHeight();

    int* blockIdsCUDA;
    int* blockOCUDA;
    float* mvpCUDA;
    cudaMalloc(&blockIdsCUDA, blockIds.size() * sizeof(int));
    cudaMalloc(&blockOCUDA, blockO.size() * sizeof(int));
    cudaMalloc(&mvpCUDA, 12 * sizeof(float));

    float* depthBuffer;
    uint32_t* renderBuffer32;
    uint32_t* counterBuffer;
    int* locks;

    cudaMalloc(&depthBuffer, width * height * sizeof(float));
    cudaMalloc(&renderBuffer32, width * height * sizeof(uint32_t) * 3);
    cudaMalloc(&counterBuffer, width * height * sizeof(uint32_t));
    cudaMalloc(&locks, width * height * sizeof(int));

    cudaMemcpy(blockIdsCUDA, blockIds.data(), blockIds.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(blockOCUDA, blockO.data(), blockO.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mvpCUDA, mvp.val, 12 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(locks, 0x00, sizeof(uint32_t) * width * height);

    int imgBlocks = (int)((width * height + threadsPerBlock - 1) / threadsPerBlock);
    initializeFloatBuffer<<<imgBlocks, threadsPerBlock>>>(depthBuffer, width * height, std::numeric_limits<float>::max());
    cudaDeviceSynchronize();

    renderBlocks<<<splatBlocks, threadsPerBlock>>>(pointCloudPosCUDA, pointCloudColCUDA, blockOffsetsCUDA, blockSizesCUDA, blockIdsCUDA, blockOCUDA, blockIds.size(), mvpCUDA, width, height, epsilon, locks, depthBuffer, renderBuffer32, counterBuffer);
    cudaDeviceSynchronize();

    cudaFree(locks);
    cudaFree(blockIdsCUDA);
    cudaFree(blockOCUDA);
    cudaFree(mvpCUDA);

    uint8_t* renderBuffer8;
    cudaMalloc(&renderBuffer8, width * height * sizeof(uint8_t) * 3);

    render32toRender8<<<imgBlocks * 3, threadsPerBlock>>>(renderBuffer8, renderBuffer32, counterBuffer, width * height);
    cudaDeviceSynchronize();

    cudaFree(renderBuffer32);
    cudaFree(counterBuffer);

    *color = renderBuffer8;
    *depth = depthBuffer;

    return splatBlocks;
}

void ProjectCloud::applyDepthFilter(const CameraCalibration &calibration, uint8_t *colorBuffer, float *depthBuffer, float **tensorPtr)
{
    std::vector<float*> imgResolutions(depthRescaleDepth + 1);
    imgResolutions[0] = depthBuffer;

    int newWidth = calibration.getWidth();
    int newHeight = calibration.getHeight();
    int nrBlocks = (int)((newWidth * newHeight + threadsPerBlock - 1) / threadsPerBlock);

    for (int i = 1; i < imgResolutions.size(); ++i)
    {
        newWidth /= 2;
        newHeight /= 2;
        nrBlocks = (int)((newWidth * newHeight + threadsPerBlock - 1) / threadsPerBlock);

        cudaMalloc(&imgResolutions[i], (newWidth * newHeight) * sizeof(float));

        reduce<<<nrBlocks, threadsPerBlock>>>(imgResolutions[i - 1], imgResolutions[i], newWidth, newHeight);
        cudaDeviceSynchronize();
    }

    for (int i = imgResolutions.size() - 1; i >= 1; --i)
    {
        uint8_t* gradMask;
        cudaMalloc(&gradMask, (newWidth * newHeight) * sizeof(uint8_t));

        laplacianKernel<<<nrBlocks, threadsPerBlock>>>(imgResolutions[i], gradMask, newWidth, newHeight);
        cudaDeviceSynchronize();

        newWidth *= 2;
        newHeight *= 2;
        nrBlocks = (int)((newWidth * newHeight + threadsPerBlock - 1) / threadsPerBlock);

        uint8_t* mask;
        cudaMalloc(&mask, newWidth * newHeight * sizeof(uint8_t));

        compareImgsKernel<<<nrBlocks, threadsPerBlock>>>(imgResolutions[i], imgResolutions[i - 1], gradMask, mask, newWidth, newHeight);
        cudaDeviceSynchronize();

        cudaFree(gradMask);

        if (i == 1)
        {
            cudaMalloc(tensorPtr, newWidth * newHeight * 5 * sizeof(float));

            float min = std::numeric_limits<float>::max(), max = std::numeric_limits<float>::min();

            float* minResult;
            float* maxResult;
            cudaMalloc(&minResult, sizeof(float));
            cudaMalloc(&maxResult, sizeof(float));

            cudaMemcpy(minResult, &min, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(maxResult, &max, sizeof(float), cudaMemcpyHostToDevice);

            findMinMax<<<nrBlocks, threadsPerBlock>>>(imgResolutions[i - 1], newWidth * newHeight, minResult, maxResult);
            cudaDeviceSynchronize();

            cudaMemcpy(&min, minResult, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&max, maxResult, sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(minResult);
            cudaFree(maxResult);

            removeMask<<<nrBlocks, threadsPerBlock>>>(imgResolutions[i - 1], colorBuffer, mask, *tensorPtr, min, max, newWidth, newHeight);
            cudaDeviceSynchronize();
        }
        else
        {
            resizeKernel<<<nrBlocks, threadsPerBlock>>>(imgResolutions[i], imgResolutions[i - 1], mask, newWidth, newHeight);
            cudaDeviceSynchronize();
        }

        cudaFree(imgResolutions[i]);
        cudaFree(mask);
    }
}

int ProjectCloud::computeFilteredRGBD(const CameraCalibration& calibration, const cv::Matx44d& extrinsics, cv::Mat* color, cv::Mat* depth)
{
    uint8_t* colorBufferPtr; float* depthBufferPtr; float* tensorPtr;
    int splatBlocks = computeRGBDInternal(calibration, extrinsics, &colorBufferPtr, &depthBufferPtr);
    applyDepthFilter(calibration, colorBufferPtr, depthBufferPtr, &tensorPtr);

    if(depth != nullptr)
    {
        cudaMemcpy(depth->ptr<float>(), depthBufferPtr, calibration.getWidth() * calibration.getHeight() * sizeof(float), cudaMemcpyDeviceToHost);
    }
    if(color != nullptr)
    {
        cudaMemcpy(color->ptr<uint8_t>(), colorBufferPtr, calibration.getWidth() * calibration.getHeight() * sizeof(uint8_t) * 3, cudaMemcpyDeviceToHost);
    }

    cudaFree(depthBufferPtr);
    cudaFree(colorBufferPtr);
    cudaFree(tensorPtr);
    return splatBlocks;
}


int ProjectCloud::computeFull(const CameraCalibration& calibration, const cv::Matx44d& extrinsics, cv::Mat* color, cv::Mat* depth)
{
    auto start = std::chrono::high_resolution_clock::now();
    uint8_t* colorBufferPtr; float* depthBufferPtr; float* tensorPtr;
    int splatBlocks = computeRGBDInternal(calibration, extrinsics, &colorBufferPtr, &depthBufferPtr);
    auto start_1 = std::chrono::high_resolution_clock::now();
    applyDepthFilter(calibration, colorBufferPtr, depthBufferPtr, &tensorPtr);
    auto start_2 = std::chrono::high_resolution_clock::now();
     //libtorch inference
    torch::Tensor tensor = torch::from_blob(
        tensorPtr, {1,5, calibration.getHeight(),calibration.getWidth()}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
    );
    tensor = tensor.to(torch::kHalf);

    torch::NoGradGuard no_grad;
    auto output = model.forward({tensor}).toTensor();
    output = output[0].permute({1, 2, 0}).contiguous();
    if(color != nullptr)
    {
        cv::Mat renderF = cv::Mat(cv::Size(calibration.getWidth(), calibration.getHeight()), CV_16FC3);
        cudaMemcpy(renderF.ptr<at::Half>(), output.data_ptr(), calibration.getWidth() * calibration.getHeight() * sizeof(at::Half) * 3, cudaMemcpyDeviceToHost);
        renderF.convertTo(*color, CV_8UC3, 255.0);
    }

    if(depth != nullptr)
    {
        cudaMemcpy(depth->ptr<float>(), depthBufferPtr, calibration.getWidth() * calibration.getHeight() * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(depthBufferPtr);
    cudaFree(colorBufferPtr);
    cudaFree(tensorPtr);

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "PROJECT_CLOUD::COMPUTE_FULL: projection[" << std::chrono::duration_cast<std::chrono::milliseconds>(start_1 - start).count() << "], filter[" << std::chrono::duration_cast<std::chrono::milliseconds>(start_2 - start_1).count() << "], unet[" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start_2).count() << "], Total[" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "]" << std::endl;

    return splatBlocks;
}

int ProjectCloud::cull(const CameraCalibration& calibration, const cv::Matx44d& extrinsics, std::vector<int> &blockIds, std::vector<int> &blockO)
{
    cv::Matx44d intrinsics = cv::Matx44d::eye();
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            intrinsics(row, col) = calibration.getIntrinsicsMatrix()(row, col);

    Frustum frustrum{ intrinsics, extrinsics, calibration.getWidth(), calibration.getHeight() };
    int splatBlocks = 0;
    for (const auto& block_pair : grid) {
        bool eyeInCurrentBlock = false;
        if (frustrum.eye[0] <= block_pair.second.bbMax.x && frustrum.eye[0] >= block_pair.second.bbMin.x
            && frustrum.eye[1] <= block_pair.second.bbMax.y && frustrum.eye[1] >= block_pair.second.bbMin.y
            && frustrum.eye[2] <= block_pair.second.bbMax.z && frustrum.eye[2] >= block_pair.second.bbMin.z)
            eyeInCurrentBlock = true;

        if (!eyeInCurrentBlock && !frustrum.checkIfBoundingBoxIsInsideFrustum(block_pair.second.bbMin, block_pair.second.bbMax))
            continue;

        for (int i = 0; i < blockIndicesMapping[block_pair.first].second; ++i) // .second is number of cuda blocks needed for this block of computation
        {
            blockIds.push_back(blockIndicesMapping[block_pair.first].first); // .first is the index of the block of computation
            blockO.push_back(i);
        }

        splatBlocks += blockIndicesMapping[block_pair.first].second;
    }

    return splatBlocks;
}


//int ProjectCloud::depthFilterComparison(cv::Matx44d extrinsics, cv::Mat* color, cv::Mat* depth = nullptr)
//{
//    Frustum frustrum{ intrinsics, extrinsics, color->cols, color->rows };
//    cv::Matx34f mvp(cv::Mat(cv::Mat(K) * cv::Mat(extrinsics)(cv::Rect(0, 0, 4, 3))));

//    std::vector<int> blockIds;
//    std::vector<int> blockO;
//    int splatBlocks = 0;

//    for (const auto& block_pair : grid) {
//        bool eyeInCurrentBlock = false;
//        if (frustrum.eye[0] <= block_pair.second.bbMax.x && frustrum.eye[0] >= block_pair.second.bbMin.x
//            && frustrum.eye[1] <= block_pair.second.bbMax.y && frustrum.eye[1] >= block_pair.second.bbMin.y
//            && frustrum.eye[2] <= block_pair.second.bbMax.z && frustrum.eye[2] >= block_pair.second.bbMin.z)
//            eyeInCurrentBlock = true;

//        if (!eyeInCurrentBlock && !frustrum.checkIfBoundingBoxIsInsideFrustum(block_pair.second.bbMin, block_pair.second.bbMax))
//            continue;

//        float distance = cv::norm(frustrum.eye -
//                                  cv::Vec3d(0.5 * (block_pair.second.bbMax.x + block_pair.second.bbMin.x),
//                                            0.5 * (block_pair.second.bbMax.y + block_pair.second.bbMin.y),
//                                            0.5 * (block_pair.second.bbMax.z + block_pair.second.bbMin.z)));

//        for (int i = 0; i < blockIndicesMapping[block_pair.first].second; ++i) // .second is number of cuda blocks needed for this block of computation
//        {
//            blockIds.push_back(blockIndicesMapping[block_pair.first].first); // .first is the index of the block of computation
//            blockO.push_back(i);
//        }

//        splatBlocks += blockIndicesMapping[block_pair.first].second;
//    }

//    if(splatBlocks == 0)
//    {
//        return splatBlocks;
//    }

//    int* blockIdsCUDA;
//    int* blockOCUDA;
//    float* mvpCUDA;
//    cudaMalloc(&blockIdsCUDA, blockIds.size() * sizeof(int));
//    cudaMalloc(&blockOCUDA, blockO.size() * sizeof(int));
//    cudaMalloc(&mvpCUDA, 12 * sizeof(float));

//    float* depthBuffer;
//    uint32_t* renderBuffer32;
//    uint32_t* counterBuffer;
//    int* locks;
//    float* tensorPtr;

//    cudaMalloc(&depthBuffer, width * height * sizeof(float));
//    cudaMalloc(&renderBuffer32, width * height * sizeof(uint32_t) * 3);
//    cudaMalloc(&counterBuffer, width * height * sizeof(uint32_t));
//    cudaMalloc(&locks, width * height * sizeof(int));

//    cudaMemcpy(blockIdsCUDA, blockIds.data(), blockIds.size() * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(blockOCUDA, blockO.data(), blockO.size() * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(mvpCUDA, mvp.val, 12 * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemset(locks, 0x00, sizeof(uint32_t) * width * height);

//    int imgBlocks = (int)((width * height + threadsPerBlock - 1) / threadsPerBlock);
//    initializeFloatBuffer<<<imgBlocks, threadsPerBlock>>>(depthBuffer, width * height, std::numeric_limits<float>::max());
//    cudaDeviceSynchronize();

//    renderBlocks<<<splatBlocks, threadsPerBlock>>>(pointCloudPosCUDA, pointCloudColCUDA, blockOffsetsCUDA, blockSizesCUDA, blockIdsCUDA, blockOCUDA, blockIds.size(), mvpCUDA, width, height, epsilon, locks, depthBuffer, renderBuffer32, counterBuffer);
//    cudaDeviceSynchronize();

//    cudaFree(locks);
//    cudaFree(blockIdsCUDA);
//    cudaFree(blockOCUDA);
//    cudaFree(mvpCUDA);

//    uint8_t* renderBuffer8;
//    cudaMalloc(&renderBuffer8, width * height * sizeof(uint8_t) * 3);

//    render32toRender8<<<imgBlocks * 3, threadsPerBlock>>>(renderBuffer8, renderBuffer32, counterBuffer, width * height);
//    cudaDeviceSynchronize();

//    uint8_t* renderBuffer82;
//    cudaMalloc(&renderBuffer82, width * height * sizeof(uint8_t) * 3);

//    cudaMemcpy(renderBuffer82, renderBuffer8, sizeof(uint8_t) * width * height * 3, cudaMemcpyDeviceToDevice);

//    cudaFree(renderBuffer32);
//    cudaFree(counterBuffer);


//    auto start_o = std::chrono::high_resolution_clock::now();
//    uint8_t* otherMask;
//    cudaMalloc(&otherMask, width * height * sizeof(uint8_t));

//    otherDepthFilter<<<imgBlocks,threadsPerBlock>>>(depthBuffer, otherMask, K(0, 0), K(1,1), K(0,2), K(1,2), width, height);
//    cudaDeviceSynchronize();

//    cudaMalloc(&tensorPtr, width * height * 5 * sizeof(float));

//    float min = std::numeric_limits<float>::max(), max = std::numeric_limits<float>::min();

//    float* minResult;
//    float* maxResult;
//    cudaMalloc(&minResult, sizeof(float));
//    cudaMalloc(&maxResult, sizeof(float));

//    cudaMemcpy(minResult, &min, sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(maxResult, &max, sizeof(float), cudaMemcpyHostToDevice);

//    findMinMax<<<imgBlocks, threadsPerBlock>>>(depthBuffer, width * height, minResult, maxResult);
//    cudaDeviceSynchronize();

//    cudaMemcpy(&min, minResult, sizeof(float), cudaMemcpyDeviceToHost);
//    cudaMemcpy(&max, maxResult, sizeof(float), cudaMemcpyDeviceToHost);

//    cudaFree(minResult);
//    cudaFree(maxResult);

//    removeMask<<<imgBlocks, threadsPerBlock>>>(tensorPtr, renderBuffer82, otherMask, tensorPtr, min, max, width, height);
//    cudaDeviceSynchronize();

//    cv::Mat myrender{cv::Size(width, height), CV_8UC3};
//    cudaMemcpy(myrender.ptr<uint8_t>(), renderBuffer82, sizeof(uint8_t) * width * height * 3, cudaMemcpyDeviceToHost);

//    cudaFree(otherMask);
//    cudaFree(renderBuffer82);
//    cudaFree(tensorPtr);
//    auto end_o = std::chrono::high_resolution_clock::now();

//    auto start_m = std::chrono::high_resolution_clock::now();
//    std::vector<float*> imgResolutions(depthRescaleDepth + 1);
//    imgResolutions[0] = depthBuffer;

//    int newWidth = width;
//    int newHeight = height;
//    int nrBlocks = (int)((newWidth * newHeight + threadsPerBlock - 1) / threadsPerBlock);

//    for (int i = 1; i < imgResolutions.size(); ++i)
//    {
//        newWidth /= 2;
//        newHeight /= 2;
//        nrBlocks = (int)((newWidth * newHeight + threadsPerBlock - 1) / threadsPerBlock);

//        cudaMalloc(&imgResolutions[i], (newWidth * newHeight) * sizeof(float));

//        reduce<<<nrBlocks, threadsPerBlock>>>(imgResolutions[i - 1], imgResolutions[i], newWidth, newHeight);
//        cudaDeviceSynchronize();
//    }
//    for (int i = imgResolutions.size() - 1; i >= 1; --i)
//    {
//        uint8_t* gradMask;
//        cudaMalloc(&gradMask, (newWidth * newHeight) * sizeof(uint8_t));

//        laplacianKernel<<<nrBlocks, threadsPerBlock>>>(imgResolutions[i], gradMask, newWidth, newHeight);
//        cudaDeviceSynchronize();

//        newWidth *= 2;
//        newHeight *= 2;
//        nrBlocks = (int)((newWidth * newHeight + threadsPerBlock - 1) / threadsPerBlock);

//        uint8_t* mask;
//        cudaMalloc(&mask, newWidth * newHeight * sizeof(uint8_t));

//        compareImgsKernel<<<nrBlocks, threadsPerBlock>>>(imgResolutions[i], imgResolutions[i - 1], gradMask, mask, newWidth, newHeight);
//        cudaDeviceSynchronize();

//        cudaFree(gradMask);

//        if (i == 1)
//        {
//            cudaMalloc(&tensorPtr, newWidth * newHeight * 5 * sizeof(float));

//            float min = std::numeric_limits<float>::max(), max = std::numeric_limits<float>::min();

//            float* minResult;
//            float* maxResult;
//            cudaMalloc(&minResult, sizeof(float));
//            cudaMalloc(&maxResult, sizeof(float));

//            cudaMemcpy(minResult, &min, sizeof(float), cudaMemcpyHostToDevice);
//            cudaMemcpy(maxResult, &max, sizeof(float), cudaMemcpyHostToDevice);

//            findMinMax<<<nrBlocks, threadsPerBlock>>>(imgResolutions[i - 1], newWidth * newHeight, minResult, maxResult);
//            cudaDeviceSynchronize();

//            cudaMemcpy(&min, minResult, sizeof(float), cudaMemcpyDeviceToHost);
//            cudaMemcpy(&max, maxResult, sizeof(float), cudaMemcpyDeviceToHost);

//            cudaFree(minResult);
//            cudaFree(maxResult);

//            removeMask<<<nrBlocks, threadsPerBlock>>>(imgResolutions[i - 1], renderBuffer8, mask, tensorPtr, min, max, newWidth, newHeight);
//            cudaDeviceSynchronize();
//        }
//        else
//        {
//            resizeKernel<<<nrBlocks, threadsPerBlock>>>(imgResolutions[i], imgResolutions[i - 1], mask, newWidth, newHeight);
//            cudaDeviceSynchronize();
//        }

//        cudaFree(imgResolutions[i]);
//        cudaFree(mask);
//    }

//    cv::Mat rgb{cv::Size(width, height), CV_8UC3};
//    cudaMemcpy(rgb.ptr<uint8_t>(), renderBuffer8, sizeof(uint8_t) * width * height * 3, cudaMemcpyDeviceToHost);

//    cudaFree(depthBuffer);
//    cudaFree(renderBuffer8);
//    cudaFree(tensorPtr);

//    auto end_m = std::chrono::high_resolution_clock::now();

//    std::cout << (end_m - start_m).count() << "   " << (end_o - start_o).count() << std::endl;

//    cv::imshow("Mine", rgb);
//    cv::imshow("Other", myrender);
//    cv::waitKey(0);

//    return splatBlocks;
//}
