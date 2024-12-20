// nvcc task01.cu -o task01

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

// CUDA Error handler to be placed around all CUDA calls
#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

#define NUM_ELEMENTS 125000000

// Grid-striding kernel for vector add
__global__ void add(uint64_t *a, uint64_t *b, uint64_t *c){
    // Determine global thread index
    

    // Kernel operation
    for (int idx = threadIdx.x + blockIdx.x*blockDim.x;
         idx < NUM_ELEMENTS;
         idx += blockDim.x*gridDim.x) {

        c[idx] = a[idx] + b[idx];
    }
}

uint64_t generate_random_64bit() {
    // Combine two 32-bit random numbers to form a 64-bit number
    uint64_t high = (uint64_t)rand(); // Generate the high 32 bits
    uint64_t low = (uint64_t)rand();  // Generate the low 32 bits

    // Shift the high part and combine with the low part
    return (high << 32) | low;
}

int main (int argc, char **argv){
    if(argc != 5) {
        printf("Usage: %s <block_size_min> <block_size_max> <grid_size_min> <grid_size_max>\n", argv[0]);
        return 1;
    }

    // Rename input
    int block_size_min = atoi(argv[1]);
    int block_size_max = atoi(argv[2]);
    int grid_size_min = atoi(argv[3]);
    int grid_size_max = atoi(argv[4]);

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
    
    // Test different block sizes (common powers of 2)
    int block_sizes_len = ceil ( log2( block_size_max / block_size_min ) ) + 1;
    int* block_sizes = new int[block_sizes_len];
    int i = 0;
    for (int threads_per_block = block_size_min;
         threads_per_block <= block_size_max;
         threads_per_block = threads_per_block*2) {
        block_sizes[i] = threads_per_block;
        ++i;
    }
    
    // Test different grid sizes
    int grid_sizes_len = ceil ( log2( grid_size_max / grid_size_min ) ) + 1;
    int* grid_sizes = new int[ grid_sizes_len ];
    i = 0;
    for (int blocks_per_grid = grid_size_min;
         blocks_per_grid <= grid_size_max;
         blocks_per_grid = blocks_per_grid*2) {
        grid_sizes[i] = blocks_per_grid;
        ++i;
    }

    // Print block and grid sizes
    printf("block_sizes = {");
    for (int i = 0; i < block_sizes_len; ++i) {
        printf(" %d ", block_sizes[i]);
    }
    printf("}\n");
    printf("grid_sizes = {");
    for (int i = 0; i < grid_sizes_len; ++i) {
        printf(" %d ", grid_sizes[i]);
    }
    printf("}\n\n");

    // Copy back result from device to host
    CUDA_CHECK  ( cudaMemcpy(   c_host,
                                c_device,
                                sizeof(uint64_t)*NUM_ELEMENTS,
                                cudaMemcpyDeviceToHost)
                );

    // free memory on GPU
    CUDA_CHECK( cudaFree(a_device));
    CUDA_CHECK( cudaFree(b_device));
    CUDA_CHECK( cudaFree(c_device));

    // free memory on host
    delete[] a_host;
    delete[] b_host;
    delete[] c_host;

   return 0;
}
