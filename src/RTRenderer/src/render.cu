#include <sys/types.h>

#include <cfloat>
#include <cstdint>
#include <cstdio>

#include "glm/fwd.hpp"
#include "render.cuh"
#include "types.h"

__device__ glm::ucvec4 unpackUCVec4(const uint64_t v) {
    return glm::ucvec4(static_cast<unsigned char>((v >> 0) & 0xFF), static_cast<unsigned char>((v >> 8) & 0xFF),
                       static_cast<unsigned char>((v >> 16) & 0xFF), static_cast<unsigned char>((v >> 24) & 0xFF));
}

__global__ void fillBuffer(uint32_t *buffer, uint32_t value, int numElements) {
    int block_id = blockIdx.x +                         // apartment number on this floor (points across)
            blockIdx.y * gridDim.x +             // floor number in this building (rows high)
            blockIdx.z * gridDim.x * gridDim.y;  // building number in this city (panes deep)

    int block_offset = block_id *                             // times our apartment number
            blockDim.x * blockDim.y * blockDim.z;  // total threads per block (people per apartment)

    int thread_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    int idx = block_offset + thread_offset;  // global person id in the entire apartment complex

    if (idx < numElements) {
        buffer[idx] = value;
    }
}

__device__ __forceinline__ float4 matmul(const float m[16], const float4 &v) {
    float4 result;
    result.x = m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3];
    result.y = m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7];
    result.z = m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11];
    result.w = m[12] * v.x + m[13] * v.y + m[14] * v.z + m[15];
    return result;
}

__device__ __forceinline__ unsigned int packuchar4(const uchar4 &v) {
    return (static_cast<unsigned int>(v.x) << 0) | (static_cast<unsigned int>(v.y) << 8) |
            (static_cast<unsigned int>(v.z) << 16) | (static_cast<unsigned int>(v.w) << 24);
}

__device__ __forceinline__ uchar4 unpackuchar4(const unsigned int v) {
    return {static_cast<unsigned char>((v >> 0) & 0xFF), static_cast<unsigned char>((v >> 8) & 0xFF),
                static_cast<unsigned char>((v >> 16) & 0xFF), static_cast<unsigned char>((v >> 24) &0xFF)};
}


__global__ void minDepthPass(uint32_t *min_depth_buffer, const float4 * vertices, const float cam_proj[16], uint2 image_size, size_t n_points)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id >= n_points) return;


    float4 p = __ldg(&vertices[global_id]);

    float4 r = matmul(cam_proj, p);
    if(r.z <= 0.0f) return;

    int u = rintf(__fdividef(r.x, r.z));
    int v = rintf(__fdividef(r.y, r.z));

    if (u < 0 || u >= image_size.x || v < 0 || v >= image_size.y) return;

    unsigned int pixID = v * image_size.x + u;

    unsigned int depth = __float_as_uint(r.z);

    unsigned int same_pixel_mask = __match_any_sync(__activemask(), pixID);
    unsigned int min_depth = __reduce_min_sync(same_pixel_mask, depth);

    bool is_closest_thread = (depth == min_depth);

    if(is_closest_thread)
    {
        atomicMin(&min_depth_buffer[pixID], min_depth);
    }
}

__global__ void accumulatePass(const float4 *vertices, const uchar4* colors, size_t n_points, uint2 image_size, const float cam_proj[16], uint32_t* min_depth_buffer, uint32_t* output_data)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= n_points) return;

    float4 p = __ldg(&vertices[global_id]);
    float4 result = matmul(cam_proj, p);

    if(result.z <= 0.0f) return;

    int u = rintf(__fdividef(result.x, result.z));
    int v = rintf(__fdividef(result.y, result.z));

    if (u < 0 || u >= image_size.x || v < 0 || v >= image_size.y) return;

    unsigned int pixID = v * image_size.x + u;
    unsigned int min_depth = __ldg(&min_depth_buffer[pixID]);

    float min_depth_val = __uint_as_float(min_depth);
    float depth_val = result.z;

    if(depth_val > min_depth_val + 0.02f)
    {
        return; // not within depth
    }

    uchar4 color = __ldg(&colors[global_id]);

    unsigned int same_pixel_mask = __match_any_sync(__activemask(), pixID);

    unsigned int r = __reduce_add_sync(same_pixel_mask, color.x);
    unsigned int g = __reduce_add_sync(same_pixel_mask, color.y);
    unsigned int b = __reduce_add_sync(same_pixel_mask, color.z);
    unsigned int count = __reduce_add_sync(same_pixel_mask, 1);

    unsigned int lane_id = threadIdx.x & 31;
    unsigned int leader_lane = __ffs(same_pixel_mask) - 1;

    if(lane_id == leader_lane)
    {
        atomicAdd(&output_data[pixID * 4 + 0], r);
        atomicAdd(&output_data[pixID * 4 + 1], g);
        atomicAdd(&output_data[pixID * 4 + 2], b);
        atomicAdd(&output_data[pixID * 4 + 3], count);
    }
}

__global__ void resolvePass(uint8_t *image, uint32_t* color_buffer, size_t count)
{
    int block_id = blockIdx.x +                         // apartment number on this floor (points across)
            blockIdx.y * gridDim.x +             // floor number in this building (rows high)
            blockIdx.z * gridDim.x * gridDim.y;  // building number in this city (panes deep)

    int block_offset = block_id *                             // times our apartment number
            blockDim.x * blockDim.y * blockDim.z;  // total threads per block (people per apartment)

    int thread_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    int id = block_offset + thread_offset;  // global person id in the entire apartment complex

    if (id > count) return;

    unsigned int c = color_buffer[id * 4 + 3];
    if(c <= 0)
    {
        image[id * 3] = (unsigned char)(0);
        image[id * 3 + 1] = (unsigned char)(0);
        image[id * 3 + 2] = (unsigned char)(0);
        return;
    }

    unsigned int r = color_buffer[id * 4];
    unsigned int g = color_buffer[id * 4 + 1];
    unsigned int b = color_buffer[id * 4 + 2];

    image[id * 3] = (unsigned char)(r/c);
    image[id * 3 + 1] = (unsigned char)(g/c);
    image[id * 3 + 2] = (unsigned char)(b/c);
}


#define IGNORED_VALUE 0x7f7fffff

__global__ void find_local_minmax_kernel(uint32_t* d_in, uint32_t* d_local_mins, uint32_t* d_local_maxes, uint32_t numElements) {
    extern __shared__ uint32_t s_min_max[];
    uint32_t* s_min = s_min_max;
    uint32_t* s_max = s_min_max + blockDim.x;

    uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t thread_min = std::numeric_limits<uint32_t>::max();
    uint32_t thread_max = std::numeric_limits<uint32_t>::min();

    while (global_idx < numElements) {
        uint32_t value = d_in[global_idx];

        if (value != IGNORED_VALUE) {
            thread_min = min(thread_min, value);
            thread_max = max(thread_max, value);
        }

        global_idx += gridDim.x * blockDim.x;
    }

    s_min[threadIdx.x] = thread_min;
    s_max[threadIdx.x] = thread_max;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s_min[threadIdx.x] = min(s_min[threadIdx.x], s_min[threadIdx.x + stride]);
            s_max[threadIdx.x] = max(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_local_mins[blockIdx.x] = s_min[0];
        d_local_maxes[blockIdx.x] = s_max[0];
    }
}

__global__ void find_overall_minmax_kernel(uint32_t* d_local_mins, uint32_t* d_local_maxes,
                                           uint32_t* d_min_out, uint32_t* d_max_out, uint32_t numBlocks) {
    extern __shared__ uint32_t s_min_max_final[];
    uint32_t* s_min = s_min_max_final;
    uint32_t* s_max = s_min_max_final + blockDim.x;

    uint32_t global_idx = threadIdx.x;

    uint32_t thread_min = std::numeric_limits<uint32_t>::max();
    uint32_t thread_max = std::numeric_limits<uint32_t>::min();

    while (global_idx < numBlocks) {
        thread_min = min(thread_min, d_local_mins[global_idx]);
        thread_max = max(thread_max, d_local_maxes[global_idx]);
        global_idx += blockDim.x;
    }

    s_min[threadIdx.x] = thread_min;
    s_max[threadIdx.x] = thread_max;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s_min[threadIdx.x] = min(s_min[threadIdx.x], s_min[threadIdx.x + stride]);
            s_max[threadIdx.x] = max(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *d_min_out = s_min[0];
        *d_max_out = s_max[0];
    }
}
