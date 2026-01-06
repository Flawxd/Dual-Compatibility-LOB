#include "gpu_compat.h"
#include <stdio.h>

// Simple GPU kernel that computes squares of indices
__global__ void hello_kernel(int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = idx * idx;  // Simple computation: square the index
    }
}

// Host function callable from C++ (and thus from Python via pybind11)
extern "C" void run_hello_kernel(int *output, int n) {
    // Device memory pointer
    int *d_output;

    // Allocate device memory
    gpuMalloc(&d_output, n * sizeof(int));

    // Launch kernel with appropriate grid/block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    hello_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, n);

    // Copy results back to host
    gpuMemcpy(output, d_output, n * sizeof(int), gpuMemcpyDeviceToHost);

    // Free device memory
    gpuFree(d_output);

    // Check for errors
    gpuError_t error = gpuGetLastError();
    if (error != gpuSuccess) {
        printf("GPU error: %s\n", gpuGetErrorString(error));
    }
}

// Simple test function that returns the GPU device count
extern "C" int get_device_count() {
    int deviceCount = 0;
    gpuGetDeviceCount(&deviceCount);
    return deviceCount;
}

// Get GPU device properties
extern "C" void get_device_info(char *name, int maxLen) {
    gpuDeviceProp prop;
    int device;
    gpuGetDevice(&device);
    gpuGetDeviceProperties(&prop, device);
    snprintf(name, maxLen, "%s", prop.name);
}
