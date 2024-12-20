#include <stdio.h>
#include <time.h>
#include <assert.h>

// CUDA Error handler to be placed around all CUDA calls
#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

#define VEC_SIZE 100000000
#define Delta 0.0001f

// task 1
// declaring vec_add kernel
// takes as input two vectors a and b as well as a size limit N
// writes elementwise addtion of a+b into vector c (c[i]=a[i]+b[i])
// check the results for numerical errors

// task 2
// use grid striding

// task 3
// measure preformance (time.h)

// monolithic kernel
__global__ void vec_add(int *a, int *b, int *c){

   // determine global thread index
   int idx = threadIdx.x + blockIdx.x*blockDim.x;

   if (idx < VEC_SIZE){
      // single vector addition
      c[idx] = a[idx] + b[idx];
   }

}

// grid stride kernel
__global__ void vec_add_gs(int *a, int *b, int *c){

   for (int idx = threadIdx.x + blockIdx.x*blockDim.x;
        idx < VEC_SIZE;
        idx += blockDim.x*gridDim.x){

      c[idx] = a[idx] + b[idx];

   }

}

int main (int argc, char **argv){

   int* a_host = new int[VEC_SIZE];
   int* b_host = new int[VEC_SIZE];
   int* c_host = new int[VEC_SIZE];

   // delete[] a_host;
   // delete[] b_host;
   // delete[] c_host;

   int *a_device, *b_device, *c_device, *c_device_gs;

   // start and end points for measuring
   // the duration in terms of clock cycles
   clock_t t_start, t_end;
   double tot_time_sec;
   double tot_time_milliseconds;


   int blocksize = 64;
   // define block size with 32 threads
   dim3 dimBlock(blocksize);
   dim3 dimGrid(ceil(VEC_SIZE/(float)blocksize));
   
// initializing a_host and b_host with each element equal to the index of the element
   for (int i=0; i<VEC_SIZE;i++) {
      a_host[i]=i;
      b_host[i]=i;
   }

// perform vector addtion on host
// measure time for vector operation
   t_start = clock();
   for (int i=0;i<VEC_SIZE;i++) c_host[i] = a_host[i] + b_host[i];
   t_end = clock();
   tot_time_sec = ((double)(t_end - t_start)) / CLOCKS_PER_SEC;
   tot_time_milliseconds = tot_time_sec * 1000;
   printf("\n vector addtion on host: %f seconds, %f milliseconds\n", tot_time_sec, tot_time_milliseconds);

// allocate memory for a, b and c on the GPU device (same size as on the host)
   CUDA_CHECK ( cudaMalloc( (void**) &a_device, sizeof(int)*VEC_SIZE ) );
   CUDA_CHECK ( cudaMalloc( (void**) &b_device, sizeof(int)*VEC_SIZE ) );
   CUDA_CHECK ( cudaMalloc( (void**) &c_device, sizeof(int)*VEC_SIZE ) );
   CUDA_CHECK ( cudaMalloc( (void**) &c_device_gs, sizeof(int)*VEC_SIZE ) );

// copy data from a_host to a_device onto the GPU
   CUDA_CHECK ( cudaMemcpy (a_device, a_host, sizeof(int)*VEC_SIZE,cudaMemcpyHostToDevice) );
   
// copy data from b_host to b_device onto the GPU
   CUDA_CHECK ( cudaMemcpy (b_device, b_host, sizeof(int)*VEC_SIZE,cudaMemcpyHostToDevice) );

// invoke vec_add kernel

// invoke kernel
   t_start = clock();
   vec_add<<<dimGrid,dimBlock>>>(a_device,b_device,c_device);
   CUDA_CHECK(cudaStreamSynchronize(0));
   t_end = clock();
   tot_time_sec = ((double)(t_end - t_start)) / CLOCKS_PER_SEC;
   tot_time_milliseconds = tot_time_sec * 1000;
   printf("\n vector addtion on device without grid striding: %f seconds, %f milliseconds\n", tot_time_sec, tot_time_milliseconds);


// invoke kernel grid stride
   t_start = clock();
   vec_add_gs<<<dimGrid,dimBlock>>>(a_device,b_device,c_device_gs);
   CUDA_CHECK(cudaStreamSynchronize(0));
   t_end = clock();
   tot_time_sec = ((double)(t_end - t_start)) / CLOCKS_PER_SEC;
   tot_time_milliseconds = tot_time_sec * 1000;
   printf("\n vector addtion on device with grid striding: %f seconds, %f milliseconds\n", tot_time_sec, tot_time_milliseconds);




// copy data from c_device on the GPU back to c_host
   CUDA_CHECK ( cudaMemcpy (c_host, c_device, sizeof(int)*VEC_SIZE,cudaMemcpyDeviceToHost) );

   //for (int i=0;i<10;i++) printf("c_host on device after gpu transfer: %d \n", c_host[i]);

// checking for correctness
//   for (int i=0;i<VEC_SIZE;i++) if (c_host[i]!=a_host[i]+b_host[i]) printf("Incorrect result at index %u differ: a_host[%u]=%u, b_host[%u]=%u, a+b=%u, c_host[%u]=%u\n",i,i,a_host[i],i,b_host[i],a_host[i]+b_host[i],i,c_host[i]);
//   printf("Finished comparing a_host and b_host.\n");

   for (int i=0;i<VEC_SIZE;i++) assert(c_host[i]-a_host[i]-b_host[i] < Delta);

// freeing memory on GPU
   CUDA_CHECK (cudaFree(a_device));
   CUDA_CHECK (cudaFree(b_device));
   CUDA_CHECK (cudaFree(c_device));

   delete[] a_host;
   delete[] b_host;
   delete[] c_host;

   return 0;
}
