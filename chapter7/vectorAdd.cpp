#include<mc_runtime_api.h>

__global__ void vectorADD(const float* A_d, const float* B_d, float* C_d, size_t NELEM) {
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = offset; i < NELEM; i += stride) {
    C_d[i] = A_d[i] + B_d[i];
    }
}

int main()
{
    int blocks=20;
    int threadsPerBlock=1024;
    int numSize=1024*1024;

    float *A_d=nullptr;
    float *B_d=nullptr;
    float *C_d=nullptr;

    float *A_h=nullptr;
    float *B_h=nullptr;
    float *C_h=nullptr;

    mcMalloc((void**)&A_d,numSize*sizeof(float));
    mcMalloc((void**)&B_d,numSize*sizeof(float));
    mcMalloc((void**)&C_d,numSize*sizeof(float));

    A_h=(float*)malloc(numSize*sizeof(float));
    B_h=(float*)malloc(numSize*sizeof(float));
    C_h=(float*)malloc(numSize*sizeof(float));

    for(int i=0;i<numSize;i++)
    {
        A_h[i]=3;
        B_h[i]=4;
        C_h[i]=0;
    }

    mcMemcpy(A_d,A_h,numSize*sizeof(float),mcMemcpyHostToDevice);
    mcMemcpy(B_d,B_h,numSize*sizeof(float),mcMemcpyHostToDevice);

    vectorADD<<<dim3(blocks),dim3(threadsPerBlock)>>>(A_d,B_d,C_d,numSize);

    mcMemcpy(C_h,C_d,numSize*sizeof(float),mcMemcpyDeviceToHost);

    mcFree(A_d);
    mcFree(B_d);
    mcFree(C_d);

    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}
