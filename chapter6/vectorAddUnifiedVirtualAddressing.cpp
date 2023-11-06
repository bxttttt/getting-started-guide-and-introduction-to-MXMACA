#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <mc_runtime_api.h>

using namespace std;

__global__ void vectorAdd(float* A_d, float* B_d, float* C_d, int N)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) C_d[i] = A_d[i] + B_d[i] + 0.0f;
}

int main(int argc, char *argv[]) {

    int n = atoi(argv[1]);
    cout << n << endl;

    size_t size = n * sizeof(float);
    mcError_t err;

    // Allocate the host vectors of A&B&C
    unsigned int flag = mcMallocHostPortable;
    float *a = NULL;
    float *b = NULL;
    float *c = NULL;
    err = mcMallocHost((void**)&a, size, flag);
    err = mcMallocHost((void**)&b, size, flag);
    err = mcMallocHost((void**)&c, size, flag);

    // Initialize the host vectors of A&B
    for (int i = 0; i < n; i++) {
        float af = rand() / double(RAND_MAX);
        float bf = rand() / double(RAND_MAX);
        a[i] = af;
        b[i] = bf;
    }

    // Launch the vector add kernel
    struct timeval t1, t2;
    int threadPerBlock = 256;
    int blockPerGrid = (n + threadPerBlock - 1)/threadPerBlock;
    printf("threadPerBlock: %d \nblockPerGrid: %d \n",threadPerBlock,blockPerGrid);
    gettimeofday(&t1, NULL);
    vectorAdd<<< blockPerGrid, threadPerBlock >>> (a, b, c, n);
    gettimeofday(&t2, NULL);
    double timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    cout << timeuse << endl;

    // Free host memory
    err = mcFreeHost(a);
    err = mcFreeHost(b);
    err = mcFreeHost(c);
    
    return 0;
}
