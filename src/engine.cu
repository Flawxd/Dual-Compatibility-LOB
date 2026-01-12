#include "gpu_compat.h"
#include <stdio.h>
//MAINLY FOR TESTING PURPOSES, IF YOU ARE CHANGING ANYTHING, YOU SHOULD TEST IT WITH THIS
__global__ void hello_kernel(int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = idx * idx;
    }
}

extern "C" void run_hello_kernel(int *output, int n) {
    int *d_output;

    gpuMalloc(&d_output, n * sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    hello_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, n);

    gpuMemcpy(output, d_output, n * sizeof(int), gpuMemcpyDeviceToHost);

    gpuFree(d_output);

    gpuError_t error = gpuGetLastError();
    if (error != gpuSuccess) {
        printf("GPU error: %s\n", gpuGetErrorString(error));
    }
}

extern "C" int get_device_count() {
    int deviceCount = 0;
    gpuGetDeviceCount(&deviceCount);
    return deviceCount;
}

extern "C" void get_device_info(char *name, int maxLen) {
    gpuDeviceProp prop;
    int device;
    gpuGetDevice(&device);
    gpuGetDeviceProperties(&prop, device);
    snprintf(name, maxLen, "%s", prop.name);
}
