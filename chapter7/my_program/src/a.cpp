//a.cpp:  
#include <mc_runtime_api.h>
#include <string.h>
extern "C"  __global__  void vector_add(int *A_d, size_t num)
{
	size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
	size_t stride = blockDim.x * gridDim.x;
	for (size_t i = offset; i < num; i += stride) {
		A_d[i]++;
	}
}
void func_a()
{
	size_t arrSize = 100;
	mcDeviceptr_t a_d;
	int *a_h = (int *)malloc(sizeof(int) * arrSize);
	memset(a_h, 0, sizeof(int) * arrSize);
	mcMalloc(&a_d, sizeof(int) * arrSize);
	mcMemcpyHtoD(a_d, a_h, sizeof(int) * arrSize);
	vector_add<<<1, arrSize>>>(reinterpret_cast<int *>(a_d), arrSize);
	mcMemcpyDtoH(a_h, a_d, sizeof(int) * arrSize);
	bool resCheck = true;
	for (int i; i < arrSize; i++) {
		if (a_h[i] != 1){
			resCheck = false;
		}
	}
	printf("vector add result: %s\n", resCheck ? "success": "fail");
	free(a_h);
	mcFree(a_d);
}

//a.h:
extern void func_a();
