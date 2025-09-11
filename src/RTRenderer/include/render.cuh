#pragma once

#include <cstdint>

#include "glm/fwd.hpp"
#include "types.h"

__global__ void vertexOrderOptimization(uint64_t *output_data, const float4 *vertices, const uchar4 *colors,
                                        const float cam_proj[16], uint2 image_size, size_t n_points);
__global__ void resolve(uint8_t *image, float *depth, const uint64_t *data, size_t count);
__global__ void fillBuffer(uint64_t *devArray, uint64_t value, int numElements);
__global__ void fillBuffer(uint32_t *devArray, uint32_t value, int numElements);
__global__ void findBlockMinMaxKernel(uint32_t *d_in, uint32_t* d_block_mins, uint32_t *d_block_maxes, size_t size);
__global__ void findAbsoluteMinMaxKernel(uint32_t* d_block_mins, uint32_t *d_block_maxes, uint32_t* d_absolute_min, uint32_t *d_absolute_max, size_t num_blocks);
__global__ void minDepthPass(uint32_t *min_depth_buffer, const float4 * vertices, const float cam_proj[16], uint2 image_size, size_t n_points);
__global__ void accumulatePass(const float4 *vertices, const uchar4* colors, size_t n_points, uint2 image_size, const float cam_proj[16], uint32_t* min_depth_buffer, uint32_t* output_data);
__global__ void resolvePass(uint8_t *image, uint32_t* color_buffer, size_t count);


__global__ void find_overall_minmax_kernel(uint32_t* d_local_mins, uint32_t* d_local_maxes,
                                           uint32_t* d_min_out, uint32_t* d_max_out, uint32_t numBlocks);
__global__ void find_local_minmax_kernel(uint32_t* d_in, uint32_t* d_local_mins, uint32_t* d_local_maxes, uint32_t numElements) ;
