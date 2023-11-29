// #include <benchmark/benchmark.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sys/time.h>
#include <iostream>
// #include <benchmark_test_config.hh>
// #include "dynamicParallelism.h"
#include <mc_runtime.h>
/** block size along */
#define BSX 64
#define BSY 4
/** maximum recursion depth */
#define MAX_DEPTH 4
/** region below which do per-pixel */
#define MIN_SIZE 32
/** subdivision factor along each axis */
#define SUBDIV 4
/** subdivision when launched from host */
#define INIT_SUBDIV 32
#define H (16 * 1024)
#define W (16 * 1024)
#define MAX_DWELL 512
using namespace std;



/** a useful function to compute the number of threads */
int __host__ __device__ divup(int x, int y) { return x / y + (x % y ? 1 : 0); }

/** a simple complex type */
struct complex {
    __host__ __device__ complex(float re, float im = 0)
    {
        this->re = re;
        this->im = im;
    }
    /** real and imaginary part */
    float re, im;
}; // struct complex

// operator overloads for complex numbers
inline __host__ __device__ complex operator+(const complex &a, const complex &b)
{
    return complex(a.re + b.re, a.im + b.im);
}
inline __host__ __device__ complex operator-(const complex &a) { return complex(-a.re, -a.im); }
inline __host__ __device__ complex operator-(const complex &a, const complex &b)
{
    return complex(a.re - b.re, a.im - b.im);
}
inline __host__ __device__ complex operator*(const complex &a, const complex &b)
{
    return complex(a.re * b.re - a.im * b.im, a.im * b.re + a.re * b.im);
}
inline __host__ __device__ float abs2(const complex &a) { return a.re * a.re + a.im * a.im; }
inline __host__ __device__ complex operator/(const complex &a, const complex &b)
{
    float invabs2 = 1 / abs2(b);
    return complex((a.re * b.re + a.im * b.im) * invabs2, (a.im * b.re - b.im * a.re) * invabs2);
} // operator/
/** find the dwell for the pixel */
__device__ int pixel_dwell(int w, int h, int max_dwell, complex cmin, complex cmax, int x, int y)
{
    complex dc = cmax - cmin;
    float fx = (float)x / w, fy = (float)y / h;
    complex c = cmin + complex(fx * dc.re, fy * dc.im);
    int dwell = 0;
    complex z = c;
    while (dwell < max_dwell && abs2(z) < 2 * 2) {
        z = z * z + c;
        dwell++;
    }
    return dwell;
} // pixel_dwell

/** binary operation for common dwell "reduction": MAX_DWELL + 1 = neutral
        element, -1 = dwells are different */
// #define NEUT_DWELL (MAX_DWELL + 1)
#define DIFF_DWELL (-1)
__device__ int same_dwell(int d1, int d2, int max_dwell)
{
    if (d1 == d2)
        return d1;
    else if (d1 == (max_dwell + 1) || d2 == (max_dwell + 1))
        return min(d1, d2);
    else
        return DIFF_DWELL;
} // same_dwell

/** evaluates the common border dwell, if it exists */
__device__ int border_dwell(int w, int h, int max_dwell, complex cmin, complex cmax, int x0, int y0,
                            int d)
{
    // check whether all boundary pixels have the same dwell
    int tid        = threadIdx.y * blockDim.x + threadIdx.x;
    int bs         = blockDim.x * blockDim.y;
    int comm_dwell = (max_dwell + 1);
    // for all boundary pixels, distributed across threads
    for (int r = tid; r < d; r += bs) {
        // for each boundary: b = 0 is east, then counter-clockwise
        for (int b = 0; b < 4; b++) {
            int x      = b % 2 != 0 ? x0 + r : (b == 0 ? x0 + d - 1 : x0);
            int y      = b % 2 == 0 ? y0 + r : (b == 1 ? y0 + d - 1 : y0);
            int dwell  = pixel_dwell(w, h, max_dwell, cmin, cmax, x, y);
            comm_dwell = same_dwell(comm_dwell, dwell, max_dwell);
        }
    } // for all boundary pixels
    // reduce across threads in the block
    __shared__ int ldwells[BSX * BSY];
    int nt = min(d, BSX * BSY);
    if (tid < nt)
        ldwells[tid] = comm_dwell;
    __syncthreads();
    for (; nt > 1; nt /= 2) {
        if (tid < nt / 2)
            ldwells[tid] = same_dwell(ldwells[tid], ldwells[tid + nt / 2], max_dwell);
        __syncthreads();
    }
    return ldwells[0];
} // border_dwell

/** the kernel to fill the image region with a specific dwell value */
__global__ void dwell_fill_k(int *dwells, int w, int x0, int y0, int d, int dwell)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < d && y < d) {
        x += x0, y += y0;
        dwells[y * w + x] = dwell;
    }
} // dwell_fill_k

/**
 * the kernel to fill in per-pixel values of the portion of the Mandelbrot set
 */
__global__ void mandelbrot_pixel_k(int *dwells, int w, int h, int max_dwell, complex cmin,
                                   complex cmax, int x0, int y0, int d)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < d && y < d) {
        x += x0, y += y0;
        dwells[y * w + x] = pixel_dwell(w, h, max_dwell, cmin, cmax, x, y);
    }
} // mandelbrot_pixel_k

/** computes the dwells for Mandelbrot image using dynamic parallelism; one block is launched per
   pixel
 @param dwells the output array
 @param w the width of the output image
 @param h the height of the output image
 @param cmin the complex value associated with the left-bottom corner of the image
 @param cmax the complex value associated with the right-top corner of the image
 @param x0 the starting x coordinate of the portion to compute
 @param y0 the starting y coordinate of the portion to compute
 @param d the size of the portion to compute (the portion is always a square)
 @param depth kernel invocation depth
 @remarks the algorithm reverts to per-pixel Mandelbrot evaluation once either maximum depth or
   minimum size is reached
 */
__global__ void mandelbrot_with_dp(int *dwells, int w, int h, int max_dwell, complex cmin,
                                   complex cmax, int x0, int y0, int d, int depth)
{
    x0 += d * blockIdx.x, y0 += d * blockIdx.y;
    int comm_dwell = border_dwell(w, h, max_dwell, cmin, cmax, x0, y0, d);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (comm_dwell != DIFF_DWELL) {
            // uniform dwell, just fill
            dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
            dwell_fill_k<<<grid, bs>>>(dwells, w, x0, y0, d, comm_dwell);
        } else if (depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE) {
            // subdivide recursively
            dim3 bs(blockDim.x, blockDim.y), grid(SUBDIV, SUBDIV);
            mandelbrot_with_dp<<<grid, bs>>>(dwells, w, h, max_dwell, cmin, cmax, x0, y0,
                                             d / SUBDIV, depth + 1);
        } else {
            // leaf, per-pixel kernel
            dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
            mandelbrot_pixel_k<<<grid, bs>>>(dwells, w, h, max_dwell, cmin, cmax, x0, y0, d);
        }
        // check_error(x0, y0, d);
    }
} // mandelbrot_with_dp

/** computes the dwells for Mandelbrot image
 @param dwells the output array
 @param w the width of the output image
 @param h the height of the output image
 @param cmin the complex value associated with the left-bottom corner of the image
 @param cmax the complex value associated with the right-top corner of the image
 */
__global__ void mandelbrot_without_dp(int *dwells, int w, int h, int max_dwell, complex cmin,
                                      complex cmax)
{
    // complex value to start iteration (c)
    int x             = threadIdx.x + blockIdx.x * blockDim.x;
    int y             = threadIdx.y + blockIdx.y * blockDim.y;
    int dwell         = pixel_dwell(w, h, max_dwell, cmin, cmax, x, y);
    dwells[y * w + x] = dwell;
}

__global__ void dwell_fill_k_null() { printf("111 \n"); } // dwell_fill_k

__global__ void mandelbrot_with_dp_cpu_perf() { dwell_fill_k_null<<<1, 1>>>(); }

__global__ void mandelbrot_without_dp_cpu_perf() { printf("222 \n"); }

struct timeval t1, t2;

static void BM_DynamicParallelism_WithDP()
{
    static char env_str[] = "DOORBELL_LISTEN=ON";
    putenv(env_str);

    // allocate memory
    int w         = W;
    int h         = H;
    int max_dwell = MAX_DWELL;

    size_t dwell_sz = w * h * sizeof(int);
    int *h_dwells, *d_dwells;
    mcMalloc((void **)&d_dwells, dwell_sz);
    h_dwells = (int *)malloc(dwell_sz);

    dim3 bs(BSX, BSY), grid(INIT_SUBDIV, INIT_SUBDIV);
    gettimeofday(&t1, NULL);
    mandelbrot_with_dp<<<grid, bs>>>(d_dwells, w, h, max_dwell, complex(-1.5, -1),
                                        complex(0.5, 1), 0, 0, w / INIT_SUBDIV, 1);
    gettimeofday(&t2, NULL);
    mcDeviceSynchronize();
    mcMemcpy(h_dwells, d_dwells, dwell_sz, mcMemcpyDeviceToHost);

    // free data
    mcFree(d_dwells);
    free(h_dwells);
    cout<<"BM_DynamicParallelism_WithDP over  "<<endl;
    double timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    cout << timeuse << endl;

}

static void BM_DynamicParallelism_WithoutDP()
{
    /* data size */
    int w         = W;
    int h         = H;
    int max_dwell = MAX_DWELL;

    size_t dwell_sz = w * h * sizeof(int);
    int *h_dwells, *d_dwells;
    mcMalloc((void **)&d_dwells, dwell_sz);
    h_dwells = (int *)malloc(dwell_sz);

    dim3 bs(64, 4), grid(divup(w, bs.x), divup(h, bs.y));
    gettimeofday(&t1, NULL);
    mandelbrot_without_dp<<<grid, bs>>>(d_dwells, w, h, max_dwell, complex(-1.5, -1),
                                        complex(0.5, 1));
    gettimeofday(&t2, NULL);
    mcDeviceSynchronize();
    mcMemcpy(h_dwells, d_dwells, dwell_sz, mcMemcpyDeviceToHost);

    // free data
    mcFree(d_dwells);
    free(h_dwells);
    cout<<"BM_DynamicParallelism_WithoutDP over"<<endl;
    double timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    cout << timeuse << endl;

}

static void BM_DynamicParallelism_WithDP_CPU_Perf()
{
    static char env_str[] = "DOORBELL_LISTEN=ON";
    putenv(env_str);

    int count = 0;
    mcGetDeviceCount(&count);

    mandelbrot_with_dp_cpu_perf<<<1, 1>>>();

    mcDeviceSynchronize();
    cout<<"BM_DynamicParallelism_WithDP_CPU_Perf over"<<endl;

}

static void BM_DynamicParallelism_WithoutDP_CPU_Perf()
{
    int count = 0;
    mcGetDeviceCount(&count);

    mandelbrot_without_dp_cpu_perf<<<1, 1>>>();

    mcDeviceSynchronize();
    cout<<"BM_DynamicParallelism_WithoutDP_CPU_Perf over"<<endl;

}



int main() {
	BM_DynamicParallelism_WithDP();
    BM_DynamicParallelism_WithoutDP();
    BM_DynamicParallelism_WithDP_CPU_Perf();
    BM_DynamicParallelism_WithoutDP_CPU_Perf();
	return 0;
}  