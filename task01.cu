// nvcc task01.cu -o task01

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

// CUDA Error handler to be placed around all CUDA calls
#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

#define NUM_ELEMENTS 125000000

// Declare host data
uint64_t a[NUM_ELEMENTS];
uint64_t b[NUM_ELEMENTS];

__global__ void kernel(){
    // Determine global thread index
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    // Kernel operation

}

uint64_t generate_random_64bit() {
    // Combine two 32-bit random numbers to form a 64-bit number
    uint64_t high = (uint64_t)rand(); // Generate the high 32 bits
    uint64_t low = (uint64_t)rand();  // Generate the low 32 bits

    // Shift the high part and combine with the low part
    return (high << 32) | low;
}

int main (int argc, char **argv){
    // Use GPU 1
    cudaSetDevice(1);
    
    // Gerenate host data
    srand((unsigned int)time(NULL));
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        a[i] = generate_random_64bit();
        b[i] = generate_random_64bit();
    }
    
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
