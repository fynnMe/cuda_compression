// nvcc task01.cu -o task01

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

// CUDA Error handler to be placed around all CUDA calls
#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

#define NUM_ELEMENTS 125000000

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

    // Initialize host data
    uint64_t* a_host = new uint64_t[NUM_ELEMENTS];
    uint64_t* b_host = new uint64_t[NUM_ELEMENTS];
    uint64_t* c_host = new uint64_t[NUM_ELEMENTS];

    // Initialize device data
    uint64_t* a_device = 0;
    uint64_t* b_device = 0;
    uint64_t* c_device = 0;

    
    // Gerenate host data
    srand((unsigned int)time(NULL));
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        a_host[i] = generate_random_64bit();
        b_host[i] = generate_random_64bit();
    }    

    // Allocate device data
    CUDA_CHECK  ( cudaMalloc((void**) &a_device, sizeof(uint64_t)*NUM_ELEMENTS));
    CUDA_CHECK  ( cudaMalloc((void**) &b_device, sizeof(uint64_t)*NUM_ELEMENTS));
    CUDA_CHECK  ( cudaMalloc((void**) &c_device, sizeof(uint64_t)*NUM_ELEMENTS));

    // Copy data from host to device
    CUDA_CHECK  ( cudaMemcpy(   a_device,
                                a_host,
                                sizeof(uint64_t)*NUM_ELEMENTS,
                                cudaMemcpyHostToDevice)
                );
    CUDA_CHECK  ( cudaMemcpy(   b_device,
                                b_host,
                                sizeof(uint64_t)*NUM_ELEMENTS,
                                cudaMemcpyHostToDevice)
                );

    // Invoke kernel

    // Copy back result from device to host

    // free memory on GPU

    // free memory on host

   return 0;
}
