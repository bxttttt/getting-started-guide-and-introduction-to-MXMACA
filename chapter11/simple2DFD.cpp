#include "../common/common.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mc_runtime.h>
#include <iostream>

#include <cassert>

#ifdef _USE_MCCL
#include <mccl.h>
#endif


/*
 * This example implements a 2D stencil computation, spreading the computation
 * across multiple GPUs. This requires communicating halo regions between GPUs
 * on every iteration of the stencil as well as managing multiple GPUs from a
 * single host application. Here, kernels and transfers are issued in
 * breadth-first order to each maca stream. Each maca stream is associated with
 * a single maca device.
 */

#define a0     -3.0124472f
#define a1      1.7383092f
#define a2     -0.2796695f
#define a3      0.0547837f
#define a4     -0.0073118f

// cnst for gpu
#define BDIMX       32
#define NPAD        4
#define NPAD2       8

// constant memories for 8 order FD coefficients
__device__ __constant__ float coef[5];

// set up fd coefficients
void setup_coef (void)
{
    const float h_coef[] = {a0, a1, a2, a3, a4};
    CHECK( mcMemcpyToSymbol( coef, h_coef, 5 * sizeof(float) ));
}

void saveSnapshotIstep(
    int istep,
    int nx,
    int ny,
    int ngpus,
    float **g_u2)
{
    float *iwave = (float *)malloc(nx * ny * sizeof(float));

    if (ngpus > 1)
    {
        unsigned int skiptop = nx * 4;
        unsigned int gsize = nx * ny / 2;

        for (int i = 0; i < ngpus; i++)
        {
            CHECK(mcSetDevice(i));
            int iskip = (i == 0 ? 0 : skiptop);
            int ioff  = (i == 0 ? 0 : gsize);
            CHECK(mcMemcpy(iwave + ioff, g_u2[i] + iskip,
                        gsize * sizeof(float), mcMemcpyDeviceToHost));

            // int iskip = (i == 0 ? nx*ny/2-4*nx : 0+4*nx);
            // int ioff  = (i == 0 ? 0 : nx*4);
            // CHECK(mcMemcpy(iwave + ioff, g_u2[i] + iskip,
            //             skiptop * sizeof(float), mcMemcpyDeviceToHost));
        }
    }
    else
    {
        unsigned int isize = nx * ny;
        CHECK(mcMemcpy (iwave, g_u2[0], isize * sizeof(float),
                          mcMemcpyDeviceToHost));
    }

    char fname[50];
    sprintf(fname, "snap_at_step_%d.data", istep);

    FILE *fp_snap = fopen(fname, "w");

    fwrite(iwave, sizeof(float), nx * ny, fp_snap);
    // fwrite(iwave, sizeof(float), nx * 4, fp_snap);
    printf("%s: nx = %d ny = %d istep = %d\n", fname, nx, ny, istep);
    fflush(stdout);
    fclose(fp_snap);

    free(iwave);
    return;
}
// 判断算力是否大于2，大于2则就支持P2P通信
inline bool isCapableP2P(int ngpus)
{
    mcDeviceProp_t prop[ngpus];
    int iCount = 0;

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(mcGetDeviceProperties(&prop[i], i));

        if (prop[i].major >= 2) iCount++;

        printf("> GPU%d: %s %s Peer-to-Peer access\n", i,
                prop[i].name, (prop[i].major >= 2 ? "supports" : "doesn't support"));
        fflush(stdout);
    }

    if(iCount != ngpus)
    {
        printf("> no enough device to run this application\n");
        fflush(stdout);
    }

    return (iCount == ngpus);
}

/*
 * enable P2P memcopies between GPUs (all GPUs must be compute capability 2.0 or
 * later (Fermi or later))
 */
inline void enableP2P (int ngpus)
{
    for (int i = 0; i < ngpus; i++)
    {
        CHECK(mcSetDevice(i));

        for (int j = 0; j < ngpus; j++)
        {
            if (i == j) continue;

            int peer_access_available = 0;
            CHECK(mcDeviceCanAccessPeer(&peer_access_available, i, j));

            if (peer_access_available) CHECK(mcDeviceEnablePeerAccess(j, 0));
        }
    }
}
// 是否支持UnifiedAddressing
inline bool isUnifiedAddressing (int ngpus)
{
    mcDeviceProp_t prop[ngpus];

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(mcGetDeviceProperties(&prop[i], i));
    }

    const bool iuva = (prop[0].unifiedAddressing && prop[1].unifiedAddressing);
    printf("> GPU%d: %s %s Unified Addressing\n", 0, prop[0].name,
           (prop[0].unifiedAddressing ? "supports" : "doesn't support"));
    printf("> GPU%d: %s %s Unified Addressing\n", 1, prop[1].name,
           (prop[1].unifiedAddressing ? "supports" : "doesn't support"));
    fflush(stdout);
    return iuva;
}
// 2GPU的结果为252,256,4,252
inline void calcIndex(int *haloStart, int *haloEnd, int *bodyStart,
                      int *bodyEnd, const int ngpus, const int iny)
{
    // for halo
    for (int i = 0; i < ngpus; i++)
    {
        if (i == 0 && ngpus == 2)
        {
            haloStart[i] = iny - NPAD2; // 260-8=252
            haloEnd[i]   = iny - NPAD; // 260-4=256

        }
        else
        {
            haloStart[i] = NPAD;
            haloEnd[i]   = NPAD2;
        }
    }

    // for body
    for (int i = 0; i < ngpus; i++)
    {
        if (i == 0 && ngpus == 2)
        {
            bodyStart[i] = NPAD; // 4
            bodyEnd[i]   = iny - NPAD2; // 260-8=252
        }
        else
        {
            bodyStart[i] = NPAD + NPAD;
            bodyEnd[i]   = iny - NPAD;
        }
    }
}
// // src_skip: 512*(260-8) 4*512 dst_skip:0  (260-4)*512
inline void calcSkips(int *src_skip, int *dst_skip, const int nx,
                      const int iny)
{
    src_skip[0] = nx * (iny - NPAD2);// 512*(260-8)
    dst_skip[0] = 0;
    src_skip[1] = NPAD * nx; // 4*512
    dst_skip[1] = (iny - NPAD) * nx; // (260-4)*512
}

// wavelet
__global__ void kernel_add_wavelet ( float *g_u2, float wavelets, const int nx,
                                     const int ny, const int ngpus)
{ // ny为iny=260，nx=512
    // global grid idx for (x,y) plane 若gpu个数为2，则
    int ipos = (ngpus == 2 ? ny - 10 : ny / 2 - 10); // ipos=250
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; // ix就是x方向上节点编号
    unsigned int idx = ipos * nx + ix; // idx=250*512+ix

    if(ix == nx / 2) g_u2[idx] += wavelets; // 这里是说ix==256时，则
}

// fd kernel function
__global__ void kernel_2dfd_last(float *g_u1, float *g_u2, const int nx,
                                 const int iStart, const int iEnd)
{
    // global to slice : global grid idx for (x,y) plane
    unsigned int ix  = blockIdx.x * blockDim.x + threadIdx.x;

    // smem idx for current point
    unsigned int stx = threadIdx.x + NPAD;
    unsigned int idx  = ix + iStart * nx;

    // shared memory for u2 with size [4+16+4][4+16+4]
    __shared__ float tile[BDIMX + NPAD2];

    const float alpha = 0.12f;

    // register for y value
    float yval[9];

    for (int i = 0; i < 8; i++) yval[i] = g_u2[idx + (i - 4) * nx];

    // to be used in z loop
    int iskip = NPAD * nx;

#pragma unroll 9
    for (int iy = iStart; iy < iEnd; iy++)
    {
        // get front3 here
        yval[8] = g_u2[idx + iskip];

        if(threadIdx.x < NPAD)
        {
            tile[threadIdx.x]  = g_u2[idx - NPAD];
            tile[stx + BDIMX]    = g_u2[idx + BDIMX];
        }

        tile[stx] = yval[4];
        __syncthreads();

        if ( (ix >= NPAD) && (ix < nx - NPAD) )
        {
            // 8rd fd operator
            float tmp = coef[0] * tile[stx] * 2.0f;

#pragma unroll
            for(int d = 1; d <= 4; d++)
            {
                tmp += coef[d] * (tile[stx - d] + tile[stx + d]);
            }

#pragma unroll
            for(int d = 1; d <= 4; d++)
            {
                tmp += coef[d] * (yval[4 - d] + yval[4 + d]);
            }

            // time dimension
            g_u1[idx] = yval[4] + yval[4] - g_u1[idx] + alpha * tmp;
        }

#pragma unroll 8
        for (int i = 0; i < 8 ; i++)
        {
            yval[i] = yval[i + 1];
        }

        // advancd on global idx
        idx  += nx;
        __syncthreads();
    }
}

__global__ void kernel_2dfd(float *g_u1, float *g_u2, const int nx,
                            const int iStart, const int iEnd)
{
    // global to line index
    unsigned int ix  = blockIdx.x * blockDim.x + threadIdx.x;

    // smem idx for current point
    unsigned int stx = threadIdx.x + NPAD;
    unsigned int idx  = ix + iStart * nx; // ix+4*512,idx表示插值的中心点坐标

    // shared memory for x dimension
    __shared__ float line[BDIMX + NPAD2];// 对于一个block，根据模板，需要的共享内存元素数量为block线程大小+NPAD*2

    // a coefficient related to physical properties
    const float alpha = 0.12f; // 关于时间步长的系数

    // register for y value
    float yval[9]; // 寄存器数组
    // 从GPU主存中获取值，这里数据由于是沿着坐标x轴排布的，所以获取主存的数据是不连续的
    for (int i = 0; i < 8; i++) yval[i] = g_u2[idx + (i - 4) * nx];

    // skip for the bottom most y value
    int iskip = NPAD * nx; // 4*512，看上面for循环，最大下标到idx+3*nx,这里多加了1

#pragma unroll 9
    for (int iy = iStart; iy < iEnd; iy++)//对y方向的数据点进行循环
    {
        // get yval[8] here
        yval[8] = g_u2[idx + iskip];//这里每次yval的最后一个数据从主存获取，其他数据最后从寄存器获取
        // 所以内存是按坐标轴的x方向上排布的
        // read halo partk // 
        if(threadIdx.x < NPAD)
        {   // 共享内存的最前最后4个数据即(0,1,2,3)和(36,37,38,39)
            line[threadIdx.x]  = g_u2[idx - NPAD]; 
            line[stx + BDIMX]    = g_u2[idx + BDIMX];
        }

        line[stx] = yval[4]; // line获取中心点的值,注意由于每个线程的yval[4]和stx都不同，所以这样可以将line[4-35]的所有数据填满
        __syncthreads();// 直到块内线程同步

        // 8rd fd operator 这里的ix>=4,ix<512-4
        if ( (ix >= NPAD) && (ix < nx - NPAD) )
        {
            // center point
            float tmp = coef[0] * line[stx] * 2.0f;

#pragma unroll
            for(int d = 1; d <= 4; d++)
            {
                tmp += coef[d] * ( line[stx - d] + line[stx + d]);
            }

#pragma unroll
            for(int d = 1; d <= 4; d++)
            {
                tmp += coef[d] * (yval[4 - d] + yval[4 + d]);
            }

            // time dimension yval[4]=gu2[idx],g_u1和g_u2和时间推进有关
            g_u1[idx] = yval[4] + yval[4] - g_u1[idx] + alpha * tmp;
        }

#pragma unroll 8 // 这里将下移一格，即沿着坐标y轴下移，进行下一层(沿着x轴为一层)
        for (int i = 0; i < 8 ; i++)
        {
            yval[i] = yval[i + 1];
        }

        // advancd on global idx
        idx  += nx; // idx+一层的点数，接着循环
        __syncthreads();
    }
}
// 程序有多个参数，第一个为要使用的GPU个数，第二个为保存哪个时间步的波场
/*
1. argv[1]:gpu数量
2. argv[2]: 每隔多少个时间步存储数据
3. argv[3]: 一共多少时间步
4. argv[4]: 每个方向上的网格数
 */
int main( int argc, char *argv[] )
{
    int ngpus=2;

    // check device count
    CHECK(mcGetDeviceCount(&ngpus));
    printf("> Number of devices available: %i\n", ngpus);

    // check p2p capability
    isCapableP2P(ngpus);
    isUnifiedAddressing(ngpus);

    //  get it from command line
    if (argc > 1)
    {
        if (atoi(argv[1]) > ngpus)
        {
            fprintf(stderr, "Invalid number of GPUs specified: %d is greater "
                    "than the total number of GPUs in this platform (%d)\n",
                    atoi(argv[1]), ngpus);
            exit(1);
        }

        ngpus  = atoi(argv[1]);
    }

    int iMovie = 100; // 这里现在表示每隔多少步存一次数据

    if(argc >= 3) iMovie = atoi(argv[2]);

    // size
    // 时间步
    int nsteps  = 1001; 
    if(argc>=4) nsteps=atoi(argv[3]);

    printf("> run with %i devices: nsteps = %i\n", ngpus, nsteps); 
    
    // x方向点数
    const int nx      = 512; 
    // y方向点数
    const int ny      = 512; 
    // 计算每个gpu上点数，这里每个线程负责所有y方向的数据点计算
    const int iny     = ny / ngpus + NPAD * (ngpus - 1); 

    size_t isize = nx * iny; // 总的数据点数
    size_t ibyte = isize * sizeof(float); // 每块总的数据字节数
#ifndef _USE_MCCL
    size_t iexchange = NPAD * nx * sizeof(float); // 交换区域的字节数
#endif

    // set up gpu card
    float *d_u2[ngpus], *d_u1[ngpus];

    for(int i = 0; i < ngpus; i++)
    {
        // set device
        CHECK(mcSetDevice(i));

        // allocate device memories // d_u1,d_u2分别存着两个时间步的数据
        CHECK(mcMalloc ((void **) &d_u1[i], ibyte));
        CHECK(mcMalloc ((void **) &d_u2[i], ibyte));

        CHECK(mcMemset (d_u1[i], 0, ibyte));
        CHECK(mcMemset (d_u2[i], 0, ibyte));
        printf("GPU %i: %.2f MB global memory allocated\n", i,
               (4.f * ibyte) / (1024.f * 1024.f) );
        setup_coef ();
    }

    // stream definition
    mcStream_t stream_halo[ngpus], stream_body[ngpus];

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(mcSetDevice(i));
        CHECK(mcStreamCreate( &stream_halo[i] ));
        CHECK(mcStreamCreate( &stream_body[i] ));
    }

    // calculate index for computation
    int haloStart[ngpus], bodyStart[ngpus], haloEnd[ngpus], bodyEnd[ngpus];
    // 根据iny进行处理 ，2GPU的结果为252,256,4,252
    calcIndex(haloStart, haloEnd, bodyStart, bodyEnd, ngpus, iny);

    int src_skip[ngpus], dst_skip[ngpus];
    // // src_skip: 512*(260-8) 4*512 dst_skip:0  (260-4)*512
    // 根据nx,iny进行处理
    if(ngpus > 1) calcSkips(src_skip, dst_skip, nx, iny); 

    // kernel launch configuration
    // block 中的线程数量
    dim3 block(BDIMX); 
    // block数量 这样的话一个线程要处理所有y向的数据。y方向被所有的GPU分块
    dim3 grid(nx / block.x); 

    // set up event for timing
    CHECK(mcSetDevice(0));
    mcEvent_t start, stop;
    CHECK (mcEventCreate(&start));
    CHECK (mcEventCreate(&stop ));
    CHECK(mcEventRecord( start, 0 ));
#ifdef _USE_MCCL
    int devs[2] = {0, 1};
    mcclComm_t comms[2];
    assert(mcclSuccess==mcclCommInitAll(comms, ngpus, devs));
#endif
    // main loop for wave propagation
    for(int istep = 0; istep < nsteps; istep++)
    {

        // save snap image
        if(istep%iMovie==0) saveSnapshotIstep(istep, nx, ny, ngpus, d_u2);

        // add wavelet only onto gpu0
        if (istep == 0) 
        {
            CHECK(mcSetDevice(0));
            kernel_add_wavelet<<<grid, block>>>(d_u2[0], 20.0, nx, iny, ngpus);
        }

        // halo part
        for (int i = 0; i < ngpus; i++)
        {
            CHECK(mcSetDevice(i));

            // compute halo 
            kernel_2dfd<<<grid, block, 0, stream_halo[i]>>>(d_u1[i], d_u2[i],
                    nx, haloStart[i], haloEnd[i]);

            // compute internal
            kernel_2dfd<<<grid, block, 0, stream_body[i]>>>(d_u1[i], d_u2[i],
                    nx, bodyStart[i], bodyEnd[i]);
        }

        /*
            ================================================================================

            ***************************使用不同的方式在GPU间交换数据****************************

            ================================================================================
        */

#ifndef _USE_MCCL 
        // exchange halo
        // src_skip: 512*(260-8) 4*512 dst_skip:0  (260-4)*512
        if (ngpus > 1) 
        {   
            // 交换两个GPU的数据注意都是d_u1的数据，即新的时间步上的数据 这里可以考虑使用mccl？
            // 这里是将gpu0的halo区域数据给gpu1的填充区域
            CHECK(mcMemcpyAsync(d_u1[1] + dst_skip[0], d_u1[0] + src_skip[0],
                                  iexchange, mcMemcpyDefault, stream_halo[0]));
            // 这里是将gpu1的halo区域数据给gpu0的填充区域
            CHECK(mcMemcpyAsync(d_u1[0] + dst_skip[1], d_u1[1] + src_skip[1],
                                  iexchange, mcMemcpyDefault, stream_halo[1]));
        }
#else
        // 使用mccl发送填充区数据
        assert(mcclSuccess == mcclGroupStart());
        for (int i = 0; i < ngpus; ++i)
        {
            mcSetDevice(i);
            int tag = (i + 1) % 2;
            mcclSend(d_u1[i] + src_skip[i], NPAD * nx, mcclFloat, tag, comms[i], stream_halo[i]);
            mcclRecv(d_u1[i] + dst_skip[tag], NPAD * nx, mcclFloat, tag, comms[i], stream_halo[i]);
        }
        assert(mcclSuccess == mcclGroupEnd());

        for (int i = 0; i < ngpus; ++i)
        {
            mcSetDevice(i);
            // it will stall host until all operations are done
            mcStreamSynchronize(stream_halo[i]);
        }
#endif
        for (int i = 0; i < ngpus; i++)
        {
            CHECK(mcSetDevice(i));
            CHECK(mcDeviceSynchronize());
            // 交换时间步的指针
            float *tmpu0 = d_u1[i];
            d_u1[i] = d_u2[i];
            d_u2[i] = tmpu0;
        }

    } // 关于istep的for循环结束

    CHECK(mcSetDevice(0));
    CHECK(mcEventRecord(stop, 0));

    CHECK(mcDeviceSynchronize());
    CHECK(mcGetLastError());

    float elapsed_time_ms = 0.0f;
    CHECK(mcEventElapsedTime(&elapsed_time_ms, start, stop));

    elapsed_time_ms /= nsteps;
    /*
    1. nsteps=30000,NCCL:845.04 MCells/s,origin:941.21 MCells/s
    2. nsteps=15000,NCCL:817.91 MCells/s,origin:935.47 MCells/s
    3. nsteps=10000,NCCL:793.62 MCells/s,origin:925.97 MCells/s
    4. nsteps=05000,NCCL:756.32 MCells/s,origin:925.32 MCells/s
    5. nsteps=02000,NCCL:599.61 MCells/s,origin:889.43 MCells/s
    6. nsteps=01000,NCCL:470.81 MCells/s,origin:802.86 MCells/s
    可见随着循环步骤数的增加，mccl通信与原有程序的速度逐渐接近
    */
    printf("gputime: %8.2fms ", elapsed_time_ms);
    printf("performance: %8.2f MCells/s\n",
           (double)nx * ny / (elapsed_time_ms * 1e3f));
    fflush(stdout);

    CHECK(mcEventDestroy(start));
    CHECK(mcEventDestroy(stop));

    // clear
    for (int i = 0; i < ngpus; i++)
    {
        CHECK(mcSetDevice(i));

        CHECK(mcStreamDestroy(stream_halo[i]));
        CHECK(mcStreamDestroy(stream_body[i]));

        CHECK(mcFree(d_u1[i]));
        CHECK(mcFree(d_u2[i]));

        // CHECK(mcDeviceReset()); // 不注释掉会mcclCommDestroy出现段错误
    }
#ifdef _USE_MCCL
    for (int i = 0; i < ngpus; ++i)
    {
        assert(mcclSuccess == mcclCommDestroy(comms[i]));
    }
#endif
    exit(EXIT_SUCCESS);
}
