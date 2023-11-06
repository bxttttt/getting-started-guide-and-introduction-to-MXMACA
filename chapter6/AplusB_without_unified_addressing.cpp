#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <mc_runtime_api.h>
#include <iostream>

__global__ void AplusB(int *ret, int a, int b) {
    ret[threadIdx.x] = a + b + threadIdx.x;
}
int main() {
    int *ret;
    mcMalloc(&ret, 1000 * sizeof(int));
    AplusB<<< 1, 1000 >>>(ret, 10, 100);
    int *host_ret = (int *)malloc(1000 * sizeof(int));
    mcMemcpy(host_ret, ret, 1000 * sizeof(int), mcMemcpyDefault);
    for(int i = 0; i < 1000; i++)
        printf("%d: A+B = %d\n", i, host_ret[i]); 
    free(host_ret);
    mcFree(ret); 
    return 0;
}
