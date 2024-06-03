#include <stdio.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include<mc_runtime_api.h>

#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

// error checking macro
#define macaCheckErrors(msg) \
  do { \
    mcError_t __err = mcGetLastError(); \
    if (__err != mcSuccess) { \
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
            msg, mcGetErrorString(__err), \
            __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - ABORTING\n"); \
        exit(1); \
    } \
  } while (0)


const int DSIZE = 1 << 26; //64MB
#define NGPUS 4

// generate different seed for random number
void initialData(float *ip, int size)
{
   time_t t;
   srand((unsigned) time(&t));

   for (int i = 0; i < size; i++)
   {
       ip[i] = (float)(rand() & 0xFF) / 10.0f;
   }

   return;
}

// vector add function: C = A + B
void cpuVectorAdd(float *A, float *B, float *C, const int N)
{
   for (int idx = 0; idx < N; idx++)
       C[idx] = A[idx] + B[idx];
}
  
// vector add kernel: C = A + B
__global__ void gpuVectorAddKernel(const float *A, const float *B, float *C, const int N){

  for (int idx = threadIdx.x+blockDim.x*blockIdx.x; idx < N; idx+=gridDim.x*blockDim.x)         // a grid-stride loop
    C[idx] = A[idx] + B[idx]; // do the vector (element) add here
}

// check results from host and gpu
void checkResult(float *hostRef, float *gpuRef, const int N)
{
   double epsilon = 1.0E-8;
   bool match = 1;
   for (int i = 0; i < N; i++)
   {
       if (abs(hostRef[i] - gpuRef[i]) > epsilon)
       {
           match = 0;
           printf("The vector-add results do not match!\n");
           printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                  gpuRef[i], i);
           break;
       }
   }
   // if (match) printf("The vector-add results match.\n\n");
   return;
}

// 程序有多个参数，第一个为要使用的GPU个数，第二个为保存哪个时间步的波场
/*
 1. argv[1]:GPU数量 (nGpus)
 2. argv[2]:线程块大小（blockSize）
 3. argv[3]:数据量（dataSize）, default is 26(1<<26=64MB)
 */
int main( int argc, char *argv[] )
{
  int nGpus;
  mcGetDeviceCount(&nGpus);
  nGpus = (nGpus > NGPUS) ? NGPUS : nGpus;
  printf("> Number of devices available: %i\n", nGpus);
  //  get it from command line
  if (argc > 1)
  {
    if (atoi(argv[1]) > nGpus)
    {
      fprintf(stderr, "Invalid number of GPUs specified: %d is greater "
                    "than the total number of GPUs in this platform (%d)\n",
                    atoi(argv[1]), nGpus);
      exit(1);
    }
    nGpus  = atoi(argv[1]);
  }
  
  // blockSize is set to 1 for slowing execution time per GPU
  int blockSize = 1; 
  // It would be faster if blockSize is set to multiples of 64(waveSize)
  if(argc >= 3) blockSize = atoi(argv[2]);
  int dataSize = DSIZE;
  if(argc >= 4) dataSize = 1 << abs(atoi(argv[3]));
  printf("> total array size is %iMB, using %i devices with each device handling %iMB\n", dataSize/1024/1024, nGpus, dataSize/1024/1024/nGpus); 
  
  float *d_A[NGPUS], *d_B[NGPUS], *d_C[NGPUS];
  float *h_A[NGPUS], *h_B[NGPUS], *hostRef[NGPUS], *gpuRef[NGPUS];
  mcStream_t stream[NGPUS];

  int iSize = dataSize / nGpus;
  size_t iBytes = iSize * sizeof(float);
  for (int i = 0; i < nGpus; i++) {
    //set current device
    mcSetDevice(i);
    
    //allocate device memory 
    mcMalloc((void **) &d_A[i], iBytes);
    mcMalloc((void **) &d_B[i], iBytes);
    mcMalloc((void **) &d_C[i], iBytes);
    
    //allocate page locked host memory for asynchronous data transfer
    mcMallocHost((void **) &h_A[i], iBytes);
    mcMallocHost((void **) &h_B[i], iBytes);
    mcMallocHost((void **) &hostRef[i], iBytes);
    mcMallocHost((void **) &gpuRef[i], iBytes);

    // initialize data at host side
    initialData(h_A[i], iSize);
    initialData(h_B[i], iSize);
    //memset(hostRef[i], 0, iBytes);
    //memset(gpuRef[i],  0, iBytes);
  }
  mcDeviceSynchronize();

  // distribute the workload across multiple devices
  unsigned long long dt = dtime_usec(0);
  for (int i = 0; i < nGpus; i++) {
    //set current device
    mcSetDevice(i);
    mcStreamCreate(&stream[i]);
     
    // transfer data from host to device
    mcMemcpyAsync(d_A[i],h_A[i], iBytes, mcMemcpyHostToDevice, stream[i]);
    mcMemcpyAsync(d_B[i],h_B[i], iBytes, mcMemcpyHostToDevice, stream[i]);
      
    // invoke kernel at host side
    dim3 block (blockSize);
    dim3 grid  (iSize/blockSize);
    gpuVectorAddKernel<<<grid,block,0,stream[i]>>>(d_A[i], d_B[i], d_C[i], iSize);
        
    // copy kernel result back to host side
    mcMemcpyAsync(gpuRef[i],d_C[i],iBytes,mcMemcpyDeviceToHost,stream[i]);
  }
  mcDeviceSynchronize();
  dt = dtime_usec(dt);
  std::cout << "> The execution time with " << nGpus <<"GPUs:  "<< dt/(float)USECPSEC << "s" << std::endl;
  
  // check the results from host and gpu devices
  for (int i = 0; i < nGpus; i++) {
    // add vector at host side for result checks
    cpuVectorAdd(h_A[i], h_B[i], hostRef[i], iSize);

    // check device results
    checkResult(hostRef[i], gpuRef[i], iSize);

    // free device global memory
    mcSetDevice(i);
    mcFree(d_A[i]);
    mcFree(d_B[i]);
    mcFree(d_C[i]);

    // free host memory
    mcFreeHost(h_A[i]);
    mcFreeHost(h_B[i]);
    mcFreeHost(hostRef[i]);
    mcFreeHost(gpuRef[i]);

    mcStreamSynchronize(stream[i]);
    mcStreamDestroy(stream[i]);
  }
  mcDeviceSynchronize();
  return 0;
}
