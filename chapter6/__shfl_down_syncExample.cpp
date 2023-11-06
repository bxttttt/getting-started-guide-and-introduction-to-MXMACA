#include <iostream>
#include <cstdlib>
#include <mc_runtime_api.h>
using namespace std;

__global__ void test_shfl_down_sync(int A[], int B[])
{
    int tid = threadIdx.x;
    int value = B[tid];
 
    value = __shfl_down_sync(0xffffffffffffffff, value, 2);
    A[tid] = value;

} 


int main()
{
    int *A,*Ad, *B, *Bd;
    int n = 64;
    int size = n * sizeof(int);
 
    // CPU端分配内存
    A = (int*)malloc(size);
    B = (int*)malloc(size);
 
    for (int i = 0; i < n; i++)
    {   
        B[i] = rand()%101;
        std::cout << B[i] << std::endl;
    }
   
    std::cout <<"----------------------------" << std::endl;
   
    // GPU端分配内存
    mcMalloc((void**)&Ad, size);
    mcMalloc((void**)&Bd, size);
    mcMemcpy(Bd, B, size, mcMemcpyHostToDevice); 
 
    // 定义kernel执行配置，（1024*1024/512）个block，每个block里面有512个线程
    dim3 dimBlock(128);
    dim3 dimGrid(1000);
 
    // 执行kernel
    test_shfl_down_sync <<<1, 64 >>> (Ad,Bd);
   
    mcMemcpy(A, Ad, size, mcMemcpyDeviceToHost);
 
    // 校验误差
    float max_error = 0.0;
    for (int i = 0; i < 64; i++)
    {
        std::cout << A[i] << std::endl;
    }
 
    cout << "max error is " << max_error << endl;
 
    // 释放CPU端、GPU端的内存
    free(A);
    free(B);   
    mcFree(Ad);
    mcFree(Bd);
 
    return 0;
}
