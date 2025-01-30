#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <time.h>

// CUDA Error handler to be placed around all CUDA calls
#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

#define NUM_ITERATIONS_PER_CONFIG 3
#define BITSIZE 11
#define ELEMENTS_PER_INT 64 / BITSIZE

// Dummy kernel to warm up GPU
__global__ void warmup_kernel()
{
    // Empty kernel, does nothing
}

// Grid-striding kernel for vector add with variable static bit size for all elements in a[i] and b[i]
__global__ void add(uint64_t *a, uint64_t *b, uint64_t *c, int num_elements) {
    for (int idx = threadIdx.x + blockIdx.x*blockDim.x;
         idx < num_elements;
         idx += blockDim.x*gridDim.x) {
        
        // Arrays based on number of elements
        int elements_per_block = blockDim.x/ELEMENTS_PER_INT;
        extern __shared__ uint64_t shared_mem[];
        uint64_t* a_block = shared_mem;
        uint64_t* b_block = &shared_mem[elements_per_block];
        uint64_t* c_block = &shared_mem[2 * elements_per_block];


        // First thread in a block initializes memory
        if (threadIdx.x == 0) {
            for (int i = 0; i < elements_per_block; ++i) {
                a_block[i] = a[i + blockIdx.x*blockDim.x];
                b_block[i] = b[i + blockIdx.x*blockDim.x];
                c_block[i] = 0;
            }
        }

        __syncthreads(); // Wait until all threads in a block reach this point

        uint64_t base_mask = (1ULL << BITSIZE) - 1;
        uint64_t position_within_uint64 = threadIdx.x % ELEMENTS_PER_INT;
        uint64_t bitmask = base_mask << (position_within_uint64 * BITSIZE);
        uint64_t a_component, b_component, c_component;
        int index_of_uint64_in_array = threadIdx.x/ELEMENTS_PER_INT;

        // Extract components
        a_component = (a_block[index_of_uint64_in_array] & bitmask) >> (position_within_uint64 * BITSIZE);
        b_component = (b_block[index_of_uint64_in_array] & bitmask) >> (position_within_uint64 * BITSIZE);

        // Perform addition
        c_component = a_component + b_component;

        // Compress
        atomicOr((unsigned long long*)&c_block[index_of_uint64_in_array], 
         (unsigned long long)(c_component << (position_within_uint64 * BITSIZE)));

        __syncthreads(); // Wait until all threads in a block reach this point
        
        // First thread in a block copies back
        if (threadIdx.x == 0) {
            for (int i = 0; i < elements_per_block; ++i) {
                c[i + blockIdx.x*blockDim.x] = c_block[i];
            }
        }
    }
}

uint64_t generate_random_64bit() {
    uint64_t high = (uint64_t)rand(); // Generate the high 32 bits
    uint64_t low = (uint64_t)rand();  // Generate the low 32 bits
    uint64_t result = (high << 32) | low;
    
    // Calculate mask for one element of given BITSIZE
    // Example: for 4-bit elements, single_element_mask = 0x0F
    uint64_t single_element_mask = (1ULL << (BITSIZE - 1)) - 1;
    
    // Calculate full mask for all elements
    uint64_t full_mask = 0;
    
    for (int i = 0; i < ELEMENTS_PER_INT; i++) {
        full_mask |= (single_element_mask << (i * BITSIZE));
    }
    
    // Apply mask to ensure MSB is 0 for each element of given BITSIZE
    return result & full_mask;
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
    if(argc != 4) {
        printf("Usage: %s <num_elements> <block_size> <grid_size>\n", argv[0]);
        return 1;
    }

    // Enable accurate printf debugging
    setbuf(stdout, NULL);

    // Open CSV
    FILE *csv_file = fopen("compressed_configs.csv", "a");
    if (csv_file == NULL) {
        printf("Error opening CSV file!\n");
        return 1;
    }

    // Write CSV header if needed
    fseek(csv_file, 0, SEEK_END);
    long size = ftell(csv_file);
    if (size == 0) {
        // File is empty, write header
        fprintf(csv_file, "array_size;block_size;grid_size;runtime\n");
    }

    // Rename input
    int num_elements = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    int grid_size = atoi(argv[3]);

    // Use GPU 1
    // cudaSetDevice(1);

    // Initialize host data
    uint64_t* a_host = new uint64_t[num_elements];
    uint64_t* b_host = new uint64_t[num_elements];
    uint64_t* c_host = new uint64_t[num_elements];

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

        // Call kernel
        cudaEventRecord(start);
        add<<<grid_size, block_size, 3*(block_size/ELEMENTS_PER_INT)>>>(a_device, b_device, c_device, num_elements);
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

        // Copy back result from device to host
        CUDA_CHECK  ( cudaMemcpy(   c_host,
                                    c_device,
                                    sizeof(uint64_t)*num_elements,
                                    cudaMemcpyDeviceToHost)
                    );
    }

    // Calculate average runtime
    avg_time_milliseconds = 0;
    for (int k = 0; k < NUM_ITERATIONS_PER_CONFIG; ++k) {
        avg_time_milliseconds += tot_time_milliseconds[k];
    }
    avg_time_milliseconds = avg_time_milliseconds / NUM_ITERATIONS_PER_CONFIG;

    // Print average runtime
    printf("num_elements: %d, block size: %d, grid_size: %d, runtime: %.6fms\n", num_elements, block_size, grid_size, avg_time_milliseconds);

    // Add csv data entries
    fprintf(csv_file, "%d;%d;%d;%.6f\n", num_elements, block_size, grid_size, avg_time_milliseconds);

    // free memory on GPU
    CUDA_CHECK( cudaFree(a_device) );
    CUDA_CHECK( cudaFree(b_device) );
    CUDA_CHECK( cudaFree(c_device) );

    // free memory on host
    delete[] a_host;
    delete[] b_host;
    delete[] c_host;

    fclose(csv_file);

    return 0;
}
