#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <mc_runtime_api.h>
#include <iostream>
using namespace std;

__global__ void assignKernel(int *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid % 2 == 0) {
        data[tid] = 20;
    } else {
        data[tid] = 10;
    }
}
int main(){
    int *a;
    a=(int *)malloc(sizeof(int)*16*16);
    int i;
    for(i=0;i<16*16;i++) a[i]=(int)rand() %10+1;
    int *da;
    mcMalloc((void **)&da,sizeof(int)*16*16);
    mcMemcpy(da,a,sizeof(int)*16*16,mcMemcpyHostToDevice);
    assignKernel<<<16,16>>>(da);
    mcMemcpy(a,da,sizeof(int)*16*16,mcMemcpyDeviceToHost);
    for(i=0;i<16*16;i++) cout<<i<<" a[i]:"<<a[i]<<endl;
    return 0;
}
