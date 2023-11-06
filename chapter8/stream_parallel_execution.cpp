#include <stdio.h>
#include <malloc.h>
#include <mc_runtime_api.h>
#define FULL_DATA_SIZE 10000
#define N 1000
#define BLOCKNUM 16
#define THREADNUM 64

__global__ void kernel(int *a,int *b,int *c){
    int idx=threadIdx.x+blockIdx.x*blockDim.x;
    if (idx<N){
        a[idx]*=a[idx];
        a[idx]+=1;
        b[idx]*=b[idx];
        b[idx]+=1;
        c[idx]*=c[idx];
        c[idx]+=1;
    }
}

int main(){
    int i;
    int *host_a,*host_b,*host_c;
    host_a=(int *)malloc(sizeof(int)*FULL_DATA_SIZE);
    host_b=(int *)malloc(sizeof(int)*FULL_DATA_SIZE);
    host_c=(int *)malloc(sizeof(int)*FULL_DATA_SIZE);
    for (i=0;i<FULL_DATA_SIZE;i++){
        host_a[i]=i;
        host_b[i]=i;
        host_c[i]=i;
    }
    int *dev0_a,*dev1_a,*dev0_b,*dev1_b,*dev0_c,*dev1_c;
    mcMalloc((int**)&dev0_a,N*sizeof(int));
    mcMalloc((int**)&dev1_a,N*sizeof(int));
    mcMalloc((int**)&dev0_b,N*sizeof(int));
    mcMalloc((int**)&dev1_b,N*sizeof(int));
    mcMalloc((int**)&dev0_c,N*sizeof(int));
    mcMalloc((int**)&dev1_c,N*sizeof(int));
    mcError_t mcStatus;
    mcStream_t stream0,stream1;
    mcStreamCreate(&stream0);
    mcStreamCreate(&stream1);
    for (i = 0; i < FULL_DATA_SIZE; i += N * 2)
    {
        mcStatus = mcMemcpyAsync(dev0_a, host_a + i, N * sizeof(int), 
                    mcMemcpyHostToDevice, stream0);
        if (mcStatus != mcSuccess)
        {
            printf("mcMemcpyAsync0 a failed!\n");
        }
    
        mcStatus = mcMemcpyAsync(dev1_a, host_a + N + i, N * sizeof(int), 
                    mcMemcpyHostToDevice, stream1);
        if (mcStatus != mcSuccess)
        {
            printf("mcMemcpyAsync1 a failed!\n");
        }
    
        mcStatus = mcMemcpyAsync(dev0_b, host_b + i, N * sizeof(int), 
                    mcMemcpyHostToDevice, stream0);
        if (mcStatus != mcSuccess)
        {
            printf("mcMemcpyAsync0 b failed!\n");
        }
    
        mcStatus = mcMemcpyAsync(dev1_b, host_b + N + i, N * sizeof(int), 
                    mcMemcpyHostToDevice, stream1);
        if (mcStatus != mcSuccess)
        {
            printf("mcMemcpyAsync1 b failed!\n");
        }
        


        kernel <<<N/BLOCKNUM, THREADNUM, 0, stream0 >>>(dev0_a, dev0_b, dev0_c);
    
        kernel <<<N/BLOCKNUM, THREADNUM, 0, stream1 >>>(dev1_a, dev1_b, dev1_c);
    
        mcStatus = mcMemcpyAsync(host_c + i, dev0_c, N * sizeof(int), 
                    mcMemcpyDeviceToHost, stream0);
        if (mcStatus != mcSuccess)
        {
            printf("mcMemcpyAsync0 c failed!\n");
        }
    
        mcStatus = mcMemcpyAsync(host_c + N + i, dev1_c, N * sizeof(int), 
                    mcMemcpyDeviceToHost, stream1);
        if (mcStatus != mcSuccess)
        {
            printf("mcMemcpyAsync1 c failed!\n");
        }
    }
    for(i=0;i<20;i++){
        printf("%d ",host_a[i]);
    }
    printf("\n");
    for(i=0;i<20;i++){
        printf("%d ",host_b[i]);
    }
    printf("\n");
    for(i=0;i<20;i++){
        printf("%d ",host_c[i]);
    }
    printf("\n");
    mcStreamSynchronize(stream1);
    mcStreamSynchronize(stream0);
    mcStreamDestroy(stream1);
    mcStreamDestroy(stream0);
    mcFree(dev0_a);
    mcFree(dev1_a);
    mcFree(dev0_b);
    mcFree(dev1_b);
    mcFree(dev0_c);
    mcFree(dev1_c);
    free(host_a);
    free(host_b);
    free(host_c);
}
