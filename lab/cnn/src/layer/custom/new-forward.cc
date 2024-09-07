#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

// TODO


	
void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
}


void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host

    // Free device memory
}


void GPUInterface::get_device_properties()
{
    // int deviceCount;
    // cudaGetDeviceCount(&deviceCount);

    // for(int dev = 0; dev < deviceCount; dev++)
    // {
    //     cudaDeviceProp deviceProp;
    //     cudaGetDeviceProperties(&deviceProp, dev);

    //     std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
    //     std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
    //     std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
    //     std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
    //     std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
    //     std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
    //     std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
    //     std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
    //     std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    // }
}
