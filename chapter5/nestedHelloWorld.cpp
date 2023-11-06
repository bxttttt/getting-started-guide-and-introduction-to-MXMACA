#include <mc_runtime.h>
#include <stdio.h>


__global__ void nestedHelloWorld(int const iSize, int iDepth) { 
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d"
            " block %d\n", iDepth, tid, blockIdx.x);
    
    // condition to stop recursive execution
    if (iSize==1) return;

    //reduce block size to half
    int nThreads = iSize >> 1;

    //thread 0 lauches child grid recursively
    if (tid == 0 && nThreads >0) {
        nestedHelloWorld<<<1, nThreads>>>(nThreads, ++iDepth);
        printf("------> nested execution depth: %d\n", iDepth);
    }
}

int main(int argc, char *argv[])
{
    // launch nestedHelloWorld
    nestedHelloWorld<<<1,8>>>(8,0);
    mcDeviceSynchronize();
    return 0;
}
