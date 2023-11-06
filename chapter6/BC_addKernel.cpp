#include <stdio.h>
#include <mc_runtime_api.h>
#include<math.h>
#include <mc_common.h>

#define ThreadsPerBlock 256
#define maxGridSize 16
__global__ void BC_addKernel(const int *a, int *r)
{
    __shared__ int cache[ThreadsPerBlock];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    // copy data to shared memory from global memory
    cache[cacheIndex] = a[tid];
    __syncthreads();

    // add these data using reduce
    for (int i = 1; i < blockDim.x; i *= 2)
    {
        int index = 2 * i * cacheIndex;
        if (index < blockDim.x)
        {
            cache[index] += cache[index + i];
        }
        __syncthreads();
    }

    // copy the result of reduce to global memory
    if (cacheIndex == 0){
        r[blockIdx.x] = cache[cacheIndex];
        printf("blockIdx.x:%d  r[blockIdx.x]:%d\n",blockIdx.x,r[blockIdx.x]);
    }
        
}

int test(int *h_a,int n){
    int *a;
    mcMalloc(&a,n*sizeof(int));
    mcMemcpy(a,h_a,n*sizeof(int),mcMemcpyHostToDevice);
    int *r;
    int h_r[maxGridSize]={0};
    mcMalloc(&r,maxGridSize*sizeof(int));
    mcMemcpy(r,h_r,maxGridSize*sizeof(int),mcMemcpyHostToDevice);
    BC_addKernel<<<ceil((double)n/ThreadsPerBlock), ThreadsPerBlock>>>(a,r);
    mcMemcpy(h_a,a,n*sizeof(int),mcMemcpyDeviceToHost);
    mcMemcpy(h_r,r,maxGridSize*sizeof(int),mcMemcpyDeviceToHost);
    mcFree(r);
    mcFree(a);
    int sum=0;
    for(int i=0;i<ceil((double)n/ThreadsPerBlock);i++){
        sum+=h_r[i];
    }
    return sum;
}

int main(){
    int n=2048;
    int *h_a=(int *)malloc(n*sizeof(int));
    int sum_cpu=0;
    for(int i=0;i<n;i++){
        h_a[i]=rand()%10;
        sum_cpu+=h_a[i];
        // h_a[i]=1;
    }
    int sum_gpu;
    sum_gpu=test(h_a,n);
    printf("sum from gpu:%d\n",sum_gpu);
    printf("sum from cpu:%d\n",sum_cpu);
    free(h_a);
    return 0;
}