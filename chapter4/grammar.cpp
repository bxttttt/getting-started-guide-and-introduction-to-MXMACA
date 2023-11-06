#include "mc_runtime_api.h"
// 字符串长度
#define SIZE 1000
// 定义设备侧的字符串变量
__device__ char dstrlist[SIZE];
// 待统计的字符，managed变量可同时被设备侧和主机侧访问
__managed__ char letters[] = {'x', 'y', 'z', 'w'};
// 演示__constant__用法，定义设备侧的字符串长度
__constant__ int dsize = SIZE;
// 使用__host__ __device__修饰可同时被主机侧和设备侧调用的函数
template<typename T, typename P>
__device__ __host__ void count_if(int *count, T *data, int start, int end, int stride, P p) {
	for(int i = start; i < end; i += stride){
		if(p(data[i])){
    // __MACA_ARCH__ 宏仅在编译设备侧代码时生效
    #ifdef __MACA_ARCH__
        // 使用原子操作保证设备侧多线程执行时的正确性
        atomicAdd(count, 1);
    #else
    	*count += 1;
    #endif
    }
  }
}
// 定义核函数
__global__ void count_xyzw(int *res) {
    // 利用内建变量gridDim, blockDim, blockIdx, threadIdx对每个线程操作的字符串进行分割
    const int start = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    // 在设备侧调用count_if
    count_if(res, dstrlist, start, dsize, stride, [=](char c){
        for(auto i: letters)
            if(i == c) return true;
        return false;
    });
}

int main(void){
    // 初始化字符串
    char test_data[SIZE];
    for(int i = 0; i < SIZE; i ++){
        test_data[i] = 'a' + i % 26;
    }
    // 拷贝字符串数据至设备侧
    mcMemcpyToSymbol(dstrlist, test_data, SIZE);
    // 开辟设备侧的计数器内存并赋值为0
    int *dcnt;
    mcMalloc(&dcnt, sizeof(int));
    int dinit = 0;
    mcMemcpy(dcnt, &dinit, sizeof(int), mcMemcpyHostToDevice);
    // 启动核函数
    count_xyzw<<<4, 64>>>(dcnt);
    // 拷贝计数器值到主机侧
    int dres;
    mcMemcpy(&dres, dcnt, sizeof(int), mcMemcpyDeviceToHost);
    // 释放设备侧开辟的内存
    mcFree(dcnt);
    printf("xyzw counted by device: %d\n", dres);

    // 在主机侧调用count_if
    int hcnt = 0;
    count_if(&hcnt, test_data, 0, SIZE, 1, [=](char c){
        for(auto i: letters)
        if(i == c) return true;
        return false;
        });
    printf("xyzw counted by host: %d\n", hcnt);
    return 0;
}
