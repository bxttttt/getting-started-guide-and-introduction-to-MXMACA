#include <stdio.h>
#include <malloc.h>
#include <mc_runtime_api.h>
// #include "device_launch_parameters.h"

__global__ void myKernel(float* devPtr, int height, int width, int pitch)
{
    int row, col;
    float *rowHead;

    for (row = 0; row < height; row++)
    {
        rowHead = (float*)((char*)devPtr + row * pitch);

        for (col = 0; col < width; col++)
        {
            printf("\t%f", rowHead[col]);// 逐个打印并自增 1
            rowHead[col]++;
        }
        printf("\n");
    }
}

int main()
{
    size_t width = 6;
    size_t height = 5;
    float *h_data, *d_data;
    size_t pitch;

    h_data = (float *)malloc(sizeof(float)*width*height);
    for (int i = 0; i < width*height; i++)
        h_data[i] = (float)i;

    printf("\n\tAlloc memory.");
    mcMallocPitch((void **)&d_data, &pitch, sizeof(float)*width, height);
    printf("\n\tPitch = %d B\n", pitch);

    printf("\n\tCopy to Device.\n");
    mcMemcpy2D(d_data, pitch, h_data, sizeof(float)*width, sizeof(float)*width, height, mcMemcpyHostToDevice);

    myKernel <<<1, 1 >>> (d_data, height, width, pitch);
    mcDeviceSynchronize();

    printf("\n\tCopy back to Host.\n");
    mcMemcpy2D(h_data, sizeof(float)*width, d_data, pitch, sizeof(float)*width, height, mcMemcpyDeviceToHost);

    for (int i = 0; i < width*height; i++)
    {
        printf("\t%f", h_data[i]);
        if ((i + 1) % width == 0)
            printf("\n");
    }

    free(h_data);
    mcFree(d_data);

    getchar();
    return 0;
}
