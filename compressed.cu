#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <time.h>

// CUDA Error handler to be placed around all CUDA calls
#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

#define NUM_ITERATIONS_PER_CONFIG 3
#define BITSIZE 8
#define COMPONENTS_PER_UINT 64 / BITSIZE

// Dummy kernel to warm up GPU
__global__ void warmup_kernel()
{
    // Empty kernel, does nothing
}

// Grid-striding kernel for vector add with variable static bit size for all elements in a[i] and b[i]
__global__ void add(uint64_t *a, uint64_t *b, uint64_t *c, int num_elements) {

    // loop for grid striding
    //   which is necessary if data is bigger than number of threads on the entire device
    // TODO how does this loop actually work?
    for (int temporal_block_id = blockIdx.x;
         temporal_block_id < num_elements / gridDim.x;
         temporal_block_id += gridDim.x) { //   max_block_size * max_blocks_per_grid_stride_for_max_block_size
                                                        //      1024        *       2*numSMs(RTX A4000)
                                                        //      1024        *       2*48
                                                        //      1024        *       96
                                                        //              98304    = max_thread_on_grid
                                                        // other_block_size * other_blocks_per_grid_stride_for_other_block_size
                                                        //        512       *       192
                                                        //        256       *       384
                                                        //        128       *       768
                                                        //         64       *      1536

        // The count of threads in the block a.k.a. `blockDim.x`
        //   divided by `COMPONENTS_PER_UINT` and then floored
        //   yields the count of components per shared uint64 array in a block
        //
        //   assume `blockDim.x` having value 32 (1024)
        //   assume `COMPONENTS_PER_UINT` having value 8
        //   the `elements_per_shared_uint64_array` would be 4 (128)
        int elements_per_shared_uint64_array = blockDim.x / COMPONENTS_PER_UINT;

        // Allocate shared memory
        extern __shared__ uint64_t shared_mem[];
        uint64_t* a_block = shared_mem;
        uint64_t* b_block = &shared_mem[elements_per_shared_uint64_array];
        uint64_t* c_block = &shared_mem[2 * elements_per_shared_uint64_array];

        // First thread in a block...
        if (threadIdx.x == 0) {

            // ... initializes shared memory
            //     by loading data from global memory
            for (int i = 0; i < elements_per_shared_uint64_array; ++i) {
                a_block[i] = a[i + temporal_block_id];
                b_block[i] = b[i + temporal_block_id];
                c_block[i] = (uint64_t)0;
            }
        }

        __syncthreads(); // Wait until all threads in a block reach this point

	// `1ULL`:
	//     00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000001
	// `1Ull << BITSIZE` assuming BITSIZE has the value eight,
	//   shifts the number one from above by eight:
	//     00000000 00000000 00000000 00000000 00000000 00000000 00000001 00000000
	// `(1ULL << BITSIZE) - 1`:
	//     00000000 00000000 00000000 00000000 00000000 00000000 00000000 11111111
        uint64_t base_mask = (1ULL << BITSIZE) - 1;

	// If `BITSIZE` has value eight, then `COMPONENTS_PER_UINT` has value "64 divided by `BITSIZE` floored" so eight
	//   neighboruing threads sharing a uint64 work on neighbouring segments of a uint64
        int position_of_component_within_uint64_in_shared_array = threadIdx.x % COMPONENTS_PER_UINT;

	// Shift base_mask
	//     00000000 00000000 00000000 00000000 00000000 00000000 00000000 11111111
	//   to the left by `position_of_component_within_uint64_in_shared_array` times `BITSIZE`
	//   e.g.                     three          times   eight     so 24
	//     00000000 00000000 00000000 00000000 11111111 00000000 00000000 00000000
        uint64_t bitmask = base_mask << (position_of_component_within_uint64_in_shared_array * BITSIZE);

	// Allocate registers for decompressed values
	//   TODO maybe move this further up to not waste time with allocation here right before saving new values to the components
	//   NOTODO can it be that a register is not big enough to
	//     to store a uint64_t? - No.
        uint64_t a_component = 0;
	uint64_t b_component = 0;
        uint64_t c_component = 0;
        int index_of_uint64_in_shared_array = threadIdx.x / COMPONENTS_PER_UINT;

        // Extract components
        // `(y_block[index_of_uint64_in_shared_array] & bitmask)`
	//       00000000 00000000 00000000 00000000 11111111 00000000 00000000 00000000
	//     & xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx abcdefgh xxxxxxxx xxxxxxxx xxxxxxxx
	//     = 00000000 00000000 00000000 00000000 abcdefgh 00000000 00000000 00000000
        // `(y_block[index_of_uint64_in_shared_array] & bitmask) >> (position_of_component_within_uint64_in_shared_array * BITSIZE)` assume
	//    `BITSIZE` has value eight and `position_of_component_within_uint64_in_shared_array` has value three
	//       00000000 00000000 00000000 00000000 abcdefgh 00000000 00000000 00000000
	//       00000000 00000000 00000000 00000000 00000000 00000000 00000000 abcdefgh
	// TODO bit shift to the right `>>` introduces only zeros on the left?
        a_component = (a_block[index_of_uint64_in_shared_array] & bitmask) >> (position_of_component_within_uint64_in_shared_array * BITSIZE);
        b_component = (b_block[index_of_uint64_in_shared_array] & bitmask) >> (position_of_component_within_uint64_in_shared_array * BITSIZE);

        // Perform addition
	//       00000000 00000000 00000000 00000000 00000000 00000000 00000000 abcdefgh
	//     + 00000000 00000000 00000000 00000000 00000000 00000000 00000000 ijklmnop
	//     = 00000000 00000000 00000000 00000000 00000000 00000000 00000000 qrstuvwz
        c_component = a_component + b_component;
        //c_component = 255;
        //c_component = 127;
        //c_component = 1;
        //c_component = c_component % 255;
        //c_component = b_component;

	// TODO maybe replace by `__threadfence();`
	__syncthreads(); // Wait until all threads in a block reach this point

	// Each thread compresses and stores it's c_component into shared memory
	//   assume `BITSIZE` has value eight and `position_of_component_within_uint64_in_shared_array` has value three
	//       00000000 00000000 00000000 00000000 00000000 00000000 00000000 qrstuvwz
	//       00000000 00000000 00000000 00000000 qrstuvwz 00000000 00000000 00000000
	// then e.g. the atomicOr(...) to write to the uint_64_t that holds the compressed values
	//       xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx 00000000 xxxxxxxx xxxxxxxx xxxxxxxx
	//    OR 00000000 00000000 00000000 00000000 qrstuvwz 00000000 00000000 00000000
	//     = xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx qrstuvwz xxxxxxxx xxxxxxxx xxxxxxxx
	// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicor
	//   "The 64-bit version of atomicOr() is only supported by devices of compute capability 5.0 and higher."
	// the RTX A4000 has compute capability 8.6
	//   according to https://developer.nvidia.com/cuda-gpus
	// and sm_xx Version can be read from
	//   https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
	// TODO problem is with the atomicOr
	//
	//   Does shifting to the left automatically introduce zeros
	//   on the right or can it introduce ones on the right?
	//   Seemingly it does only introduce zeros to the right.
/*
        atomicOr(
	  (unsigned long long*)&c_block[index_of_uint64_in_shared_array],
          (unsigned long long)(c_component << (position_of_component_within_uint64_in_shared_array * BITSIZE))
          //(unsigned long long)9833521311817092772
          //(unsigned long long)id
          //(unsigned long long)index_of_uint64_in_shared_array
          //(unsigned long long)(position_of_component_within_uint64_in_shared_array << (position_of_component_within_uint64_in_shared_array * BITSIZE))
          //(unsigned long long)(COMPONENTS_PER_UINT << (position_of_component_within_uint64_in_shared_array * BITSIZE))
	);

	// should in our case write the same value as atomicOr(...)
        atomicAdd(
	  (unsigned long long*)&c_block[index_of_uint64_in_shared_array],
          (unsigned long long)(c_component << (position_of_component_within_uint64_in_shared_array * BITSIZE))
	);

	// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-fetch-or-and-nv-atomic-or
	//   https://en.cppreference.com/w/cpp/atomic/memory_order
	//   https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes
        __nv_atomic_or(
	    &c_block[index_of_uint64_in_shared_array],
            (c_component << (position_of_component_within_uint64_in_shared_array * BITSIZE)),
	    std::memory_order_acq_rel,
	    cuda::thread_scope_block
	);
*/

/*
	if (position_of_component_within_uint64_in_shared_array == 0) {
	    c_block[index_of_uint64_in_shared_array] = c_component << (position_of_component_within_uint64_in_shared_array * BITSIZE);
	}
*/


	// TODO maybe replace by `__threadfence();`
	__syncthreads(); // Wait until all threads in a block reach this point

	// TODO LoL that is so god damn inefficient
	//   the idea that threads in one grid-stride
	//   want to write to the same uint64_t is
	//   absoloutely not ideal
        atomicOr(
          (unsigned long long*)&c_block[index_of_uint64_in_shared_array],
          (unsigned long long)(c_component << (position_of_component_within_uint64_in_shared_array * BITSIZE))
        );

        __syncthreads(); // Wait until all threads in a block reach this point

        // First thread in a block reads from shared memory and
        //   writes to global memory
        if (threadIdx.x == 0) {
            for (int i = 0; i < elements_per_shared_uint64_array; ++i) {
                c[i + temporal_block_id] = c_block[i];
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

    for (int i = 0; i < COMPONENTS_PER_UINT; i++) {
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
    uint64_t* c_comp = new uint64_t[num_elements];
    uint64_t* a_gpu_dump = new uint64_t[num_elements];
    uint64_t* b_gpu_dump = new uint64_t[num_elements];


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

        // Perform addition on CPU for comparison
        for (int l = 0; l < num_elements; ++l) {
            c_comp[l] = a_host[l] + b_host[l];
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
        add<<<grid_size, block_size, 3*(block_size/COMPONENTS_PER_UINT)>>>(a_device, b_device, c_device, num_elements);
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
        CUDA_CHECK  ( cudaMemcpy(   a_gpu_dump,
                                    a_device,
                                    sizeof(uint64_t)*num_elements,
                                    cudaMemcpyDeviceToHost)
                    );
         CUDA_CHECK  ( cudaMemcpy(  b_gpu_dump,
                                    b_device,
                                    sizeof(uint64_t)*num_elements,
                                    cudaMemcpyDeviceToHost)
                    );
          CUDA_CHECK  ( cudaMemcpy( c_host,
                                    c_device,
                                    sizeof(uint64_t)*num_elements,
                                    cudaMemcpyDeviceToHost)
                    );

        // Confirm that GPU computed correctly
        for (int l = 0; l < num_elements; ++l) {
            if (c_host[l] != c_comp[l]) {
                printf("Result %d of GPU is not the same as CPU result:\n", l);
                printf("%llu + %llu = %llu (host)\n", (unsigned long long int)a_host[l], (unsigned long long int)b_host[l], (unsigned long long int)c_comp[l]);
                printf("%llu + %llu = %llu (device)\n", (unsigned long long int)a_gpu_dump[l], (unsigned long long int)b_gpu_dump[l], (unsigned long long int)c_host[l]);

		print_binary(c_comp[l]);
		print_binary(c_host[l]);
                return 1;
            }
        }
        printf("Result of GPU is the same as CPU result.\n");
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
