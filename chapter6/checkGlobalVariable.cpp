#include <mc_runtime_api.h>
#include <stdio.h>

__device__ float devData;
__global__ void checkGlobalVariable(){
    printf("Device: the value of the global variable is %f\n", devData);
    devData += 2.0;
}

int main(){
    float value = 3.14f;
    mcMemcpyToSymbol(devData, &value, sizeof(float));
    printf("Host: copy %f to the global variable\n", value);
    checkGlobalVariable<<<1,1>>>();
    mcMemcpyFromSymbol(&value, devData, sizeof(float));
    printf("Host: the value changed by the kernel to %f\n", value);
    mcDeviceReset();
    return EXIT_SUCCESS;
}
