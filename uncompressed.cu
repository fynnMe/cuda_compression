// nvcc uncompressed.cu -o uncompressed
// ./uncompressed 32 1024 1 1024

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <time.h>

// CUDA Error handler to be placed around all CUDA calls
#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

#define NUM_ITERATIONS_PER_CONFIG 3

// Dummy kernel to warm up GPU
__global__ void warmup_kernel()
{
    // Empty kernel, does nothing
}

// Grid-striding kernel for vector add
__global__ void add(uint64_t *a, uint64_t *b, uint64_t *c, int num_elements){
    for (int idx = threadIdx.x + blockIdx.x*blockDim.x;
         idx < num_elements;
         idx += blockDim.x*gridDim.x) {

        c[idx] = a[idx] + b[idx];
    }
}

uint64_t generate_random_64bit() {
    uint64_t high = (uint64_t)rand(); // Generate the high 32 bits
    uint64_t low = (uint64_t)rand();  // Generate the low 32 bits

    // Shift the high part and combine with low part
    // Use & 0x7FFFFFFFFFFFFFFF to force MSB to 0
    return ((high << 32) | low) & 0x7FFFFFFFFFFFFFFF;
}

// Print 64 bits, starting from MSB
void print_binary(uint64_t num) {
    for(int i = 63; i >= 0; i--) {
        printf("%lu", (num >> i) & 1UL);
        
        // Add space every 8 bits for readability
        if (i % 8 == 0) {
            printf(" ");
        }
    }
    printf("\n");
}

int main (int argc, char **argv){
    if(argc != 6) {
        printf("Usage: %s <num_elements> <block_size_min> <block_size_max> <grid_size_min> <grid_size_max>\n", argv[0]);
        return 1;
    }

    // Enable accurate printf debugging
    setbuf(stdout, NULL);
    FILE *csv_file_configs = fopen("uncompressed_configs.csv", "w");
    if (csv_file_configs == NULL) {
        printf("Error opening CSV file!\n");
        return 1;
    }
    FILE *csv_file_runtime = fopen("uncompressed_runtimes.csv", "a");
    if (csv_file_runtime == NULL) {
        printf("Error opening CSV file!\n");
        return 1;
    }
    // Write CSV header
    fprintf(csv_file_configs, "block_size,grid_size,runtime\n");

    // Rename input
    int num_elements = atoi(argv[1]);
    int block_size_min = atoi(argv[2]);
    int block_size_max = atoi(argv[3]);
    int grid_size_min = atoi(argv[4]);
    int grid_size_max = atoi(argv[5]);

    // Use GPU 1
    cudaSetDevice(1);

    // Initialize host data
    uint64_t* a_host = new uint64_t[num_elements];
    uint64_t* b_host = new uint64_t[num_elements];
    uint64_t* c_host = new uint64_t[num_elements];
    uint64_t* c_comp = new uint64_t[num_elements];

    // Initialize device data
    uint64_t* a_device = 0;
    uint64_t* b_device = 0;
    uint64_t* c_device = 0;

    // Allocate device data
    CUDA_CHECK  ( cudaMalloc((void**) &a_device, sizeof(uint64_t)*num_elements) );
    CUDA_CHECK  ( cudaMalloc((void**) &b_device, sizeof(uint64_t)*num_elements) );
    CUDA_CHECK  ( cudaMalloc((void**) &c_device, sizeof(uint64_t)*num_elements) );

    // Declare time measurment variables
    cudaEvent_t start, stop;
    float tot_time_milliseconds[NUM_ITERATIONS_PER_CONFIG];
    float avg_time_milliseconds;

    // Create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
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

    // Invoke dummy kernel for GPU warmup
    warmup_kernel<<<1, 1>>>();
    cudaError_t error = cudaGetLastError(); // Check for launch errors
    if (error != cudaSuccess) {
        printf("Launch error: %s\n", cudaGetErrorString(error));
        return error;
    }

    error = cudaDeviceSynchronize(); // Check for execution errors
    if (error != cudaSuccess) {
        printf("Execution error: %s\n", cudaGetErrorString(error));
        return error;
    }

    // Invoke kernel
    for (int i = 0; i < block_sizes_len; ++i) {
        for (int j = 0; j < grid_sizes_len; ++j) {
            for (int k = 0; k < NUM_ITERATIONS_PER_CONFIG; ++k) {                
                // Gerenate host data
                srand((unsigned int)time(NULL));
                for (int l = 0; l < num_elements; ++l) {
                    a_host[l] = generate_random_64bit();
                    b_host[l] = generate_random_64bit();
                }

                // Copy data from host to device
                CUDA_CHECK  ( cudaMemcpy(   a_device,
                                            a_host,
                                            sizeof(uint64_t)*num_elements,
                                            cudaMemcpyHostToDevice)
                            );
                CUDA_CHECK  ( cudaMemcpy(   b_device,
                                            b_host,
                                            sizeof(uint64_t)*num_elements,
                                            cudaMemcpyHostToDevice)
                            );
                
                // Set dim_block and dim_grid
                dim3 dim_block(block_sizes[i]);
                dim3 dim_grid(grid_sizes[i]);

                // Call kernel
                cudaEventRecord(start);
                add<<<dim_grid, dim_block>>>(a_device, b_device, c_device, num_elements);
                cudaEventRecord(stop);
                error = cudaGetLastError(); // Check for launch errors
                if (error != cudaSuccess) {
                    printf("Launch error: %s\n", cudaGetErrorString(error));
                    return error;
                }

                error = cudaDeviceSynchronize(); // Check for execution errors
                if (error != cudaSuccess) {
                    printf("Execution error: %s\n", cudaGetErrorString(error));
                    return error;
                }
                cudaEventSynchronize(stop); // Wait for the stop event to complete
                cudaEventElapsedTime(&tot_time_milliseconds[k], start, stop);
            }

            // Calculate average runtime
            avg_time_milliseconds = 0;
            for (int k = 0; k < NUM_ITERATIONS_PER_CONFIG; ++k) {
                avg_time_milliseconds += tot_time_milliseconds[k];
            }
            avg_time_milliseconds = avg_time_milliseconds / NUM_ITERATIONS_PER_CONFIG;

            // Print average runtime
            printf("num_elements: %d, block size: %d, grid_size: %d, runtime %.6fms\n", num_elements, block_sizes[i], grid_sizes[j], avg_time_milliseconds);

            // Add csv data entries
            fprintf(csv_file_configs, "%d,%d,%.6f\n", block_sizes[i], grid_sizes[j], avg_time_milliseconds);
            fprintf(csv_file_runtime, "%d,%.6f\n", num_elements, avg_time_milliseconds);
        }
    }

    // Copy back result from device to host
    CUDA_CHECK  ( cudaMemcpy(   c_host,
                                c_device,
                                sizeof(uint64_t)*num_elements,
                                cudaMemcpyDeviceToHost)
                );

    // free memory on GPU
    CUDA_CHECK( cudaFree(a_device) );
    CUDA_CHECK( cudaFree(b_device) );
    CUDA_CHECK( cudaFree(c_device) );

    // free memory on host
    delete[] a_host;
    delete[] b_host;
    delete[] c_host;
    delete[] c_comp;
    
    fclose(csv_file_configs);
    fclose(csv_file_runtime);

    return 0;
}
