#include <stdio.h>
#include <mc_common.h>
#include <mc_runtime_api.h> 

__global__ void helloFromGpu (void)
{
    printf("Hello World from GPU!\n");
}

int main(void)
{
    printf("Hello World from CPU!\n");
    
    helloFromGpu <<<1, 10>>>();
    mcDeviceReset();
    //mcDeviceReset()用来显式销毁并清除与当前设备有关的所有资源。
	return 0;
}
