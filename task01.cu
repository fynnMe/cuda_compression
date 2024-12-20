// nvcc task01.cu -o task01

#include <stdio.h>
#include <time.h>
#include <assert.h>

// CUDA Error handler to be placed around all CUDA calls
#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

__global__ void kernel(){
    // Determine global thread index
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    // Kernel operation

}

int main (int argc, char **argv){
    // Declare host data

    // Declare device data

    // Initialize device data

    // Allocate device data

    // Copy data from host to device

    // Invoke kernel

    // Copy back result from device to host

    // free memory on GPU

    // free memory on host

   return 0;
}
