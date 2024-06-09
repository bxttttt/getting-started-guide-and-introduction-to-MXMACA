/*
 * 9.4.1: 1) lots of short kernels launched asynchronously 
 * 9.4.1 {Sample#2} lots of short kernels launched asynchronously 
 * Usage:
 *   1) compiling: mxcc -x maca shortKernelsAsyncLaunch.cpp -o shortKernelsAsyncLaunch
 *   2) running：./shortKernelsAsyncLaunch
 */
#include <iostream>
#include <vector>
#include "mc_runtime.h"

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


#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start){
  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

#define N 400000 // tuned until kernel takes a few microseconds
__global__ void shortKernel(float * out_d, float * in_d){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<N) out_d[idx]=1.23*in_d[idx];
}

#define NSTEP 2000
#define NKERNEL 200
typedef float ft;

int main(){
  ft *d_input, *d_output;
  mcStream_t stream;
  mcStreamCreate(&stream);

  // device allocations
  mcMalloc(&d_input, N*sizeof(ft));
  mcMalloc(&d_output,  N*sizeof(ft));
  
  int blocks = 1;
  int threads = 64;  
  // warm-up: copy kernel image to device
  shortKernel<<<blocks, threads>>>(d_output, d_input);
  macaCheckErrors("kernel launch failure");
  mcDeviceSynchronize();
  macaCheckErrors("kernel execution failure");
  // run on device and measure execution time
  unsigned long long dt = dtime_usec(0);
  dt = dtime_usec(0);
  for(int istep=0; istep<NSTEP; istep++){
    for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
      shortKernel<<<blocks, threads, 0, stream>>>(d_output, d_input);
    }
  }
  mcStreamSynchronize(stream);

  macaCheckErrors("kernel execution failure");
  dt = dtime_usec(dt);
  std::cout << "Kernel execution time: total=" << dt/(float)USECPSEC << "s, perKernelInAvg=" << 1000*1000*dt/NKERNEL/NSTEP/(float)USECPSEC << "us." << std::endl;
  return 0;
}