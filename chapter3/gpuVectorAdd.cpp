#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <mc_runtime_api.h>

using namespace std;

// 要用 __global__ 来修饰。
// 输入指向3段显存的指针名。
__global__ void gpuVectorAddKernel(float* A_d,float* B_d,float* C_d, int N)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) C_d[i] = A_d[i] + B_d[i];
}

int main(int argc, char *argv[]) {

    int n = atoi(argv[1]);
    cout << n << endl;

    size_t size = n * sizeof(float);

    // host memery
    float *a = (float *)malloc(size);
    float *b = (float *)malloc(size);
    float *c = (float *)malloc(size);

    for (int i = 0; i < n; i++) {
        float af = rand() / double(RAND_MAX);
        float bf = rand() / double(RAND_MAX);
        a[i] = af;
        b[i] = bf;
    }

    // 定义空指针。
    float *da = NULL;
    float *db = NULL;
    float *dc = NULL;

    // 申请显存，da 指向申请的显存，注意 mcMalloc 函数传入指针的指针 (指向申请得到的显存的指针)。
    mcMalloc((void **)&da, size);
    mcMalloc((void **)&db, size);
    mcMalloc((void **)&dc, size);

    // 把内存的东西拷贝到显存，也就是把 a, b, c 里面的东西拷贝到 d_a, d_b, d_c 中。
    mcMemcpy(da,a,size,mcMemcpyHostToDevice);
    mcMemcpy(db,b,size,mcMemcpyHostToDevice);

    struct timeval t1, t2;

    // 计算线程块和网格的数量。
    int threadPerBlock = 256;
    int blockPerGrid = (n + threadPerBlock - 1)/threadPerBlock;
    printf("threadPerBlock: %d \nblockPerGrid: %d\n", threadPerBlock,blockPerGrid);

    gettimeofday(&t1, NULL);

    // 调用核函数。
    gpuVectorAddKernel<<< blockPerGrid, threadPerBlock >>> (da, db, dc, n);

    gettimeofday(&t2, NULL);

    mcMemcpy(c,dc,size,mcMemcpyDeviceToHost);

    // for (int i = 0; i < 10; i++) 
    //     cout<<vecA[i]<<" "<<vecB[i]<<" "<<vecC[i]<< endl;

double timeuse = (t2.tv_sec - t1.tv_sec) + 
(double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    cout << timeuse << endl;

    mcFree(da);
    mcFree(db);
    mcFree(dc);

    free(a);
    free(b);
    free(c);
    return 0;
}
