#ifndef GPU_COMPAT_H
#define GPU_COMPAT_H

#ifdef USE_HIP
    #include <hip/hip_runtime.h>

    #ifndef __global__
        #define __global__ __global__
    #endif

    #ifndef __device__
        #define __device__ __device__
    #endif

    #ifndef __host__
        #define __host__ __host__
    #endif

#else
    #include <cuda_runtime.h>

#endif

#ifdef USE_HIP
    #define gpuError_t hipError_t
    #define gpuSuccess hipSuccess
    #define gpuGetErrorString hipGetErrorString
    #define gpuGetLastError hipGetLastError
    #define gpuDeviceSynchronize hipDeviceSynchronize
    #define gpuMalloc hipMalloc
    #define gpuFree hipFree
    #define gpuMemcpy hipMemcpy
    #define gpuMemcpyHostToDevice hipMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
    #define gpuMemset hipMemset
    #define gpuGetDeviceCount hipGetDeviceCount
    #define gpuGetDevice hipGetDevice
    #define gpuGetDeviceProperties hipGetDeviceProperties
    #define gpuDeviceProp hipDeviceProp_t
#else
    #define gpuError_t cudaError_t
    #define gpuSuccess cudaSuccess
    #define gpuGetErrorString cudaGetErrorString
    #define gpuGetLastError cudaGetLastError
    #define gpuDeviceSynchronize cudaDeviceSynchronize
    #define gpuMalloc cudaMalloc
    #define gpuFree cudaFree
    #define gpuMemcpy cudaMemcpy
    #define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
    #define gpuMemset cudaMemset
    #define gpuGetDeviceCount cudaGetDeviceCount
    #define gpuGetDevice cudaGetDevice
    #define gpuGetDeviceProperties cudaGetDeviceProperties
    #define gpuDeviceProp cudaDeviceProp
#endif

#define checkGpuError(call) { \
    gpuError_t err = call; \
    if (err != gpuSuccess) { \
        fprintf(stderr, "GPU Error in %s:%d - %s\n", __FILE__, __LINE__, \
                gpuGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#endif
