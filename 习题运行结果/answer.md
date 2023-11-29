# new answer

## Chapter 2

### Exercise 1

#### 参考代码

```c
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
  return 0;
}
```

#### 编译结果

函数mcDeviceReset()用来显式销毁并清除与当前设备有关的所有资源。

当重置函数移除，编译运行则只输出

```
Hello World from CPU!
```

当printf在gpu上被调用，mcDeviceReset()函数使这些来自gpu的输出发送到主机，然后在控制台输出。

没有调用cudaDeviceReset()函数就不能保证这些可以被显示。

### Exercise 2

#### 参考代码

```c
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
    mcDeviceSynchronize();
	return 0;
}

```

#### 编译结果

```
Hello World from CPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
```

输出效果和helloFromGpu.c一样。

mcDeviceSynchronize()也可以用来使gpu的输出打印在用户可见控制台。

### Exercise 3

#### 参考代码

```c
#include <stdio.h>
#include <mc_common.h>
#include <mc_runtime_api.h> 

__global__ void helloFromGpu (void)
{
    if (threadIdx.x==9) printf("Hello World from GPU Thread 9!\n");
}
int main(void)
{
    printf("Hello World from CPU!\n")；
    helloFromGpu <<<1, 10>>>();
    mcDeviceReset();
    return 0;
}
```

## Chapter 3

### Exercise 1 

#### 参考代码

```c++
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <mc_runtime_api.h>

using namespace std;

// 要用 __global__ 来修饰。
// 输入指向3段显存的指针名。
__global__ void gpuVectorAddKernel(float* A_d,float* B_d,float* C_d, int N)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    // printf("threadIdx.x:%d  blockDim.x:%d  blockIdx.x:%d\n",threadIdx.x,blockDim.x,blockIdx.x);
    if (i < N) C_d[i] = A_d[i] + B_d[i];
}

int main(int argc, char *argv[]) {

    int n = 2048;
    cout << n << endl;

    size_t size = n * sizeof(float);

    // host memery
    float *a = (float *)malloc(size);
    float *b = (float *)malloc(size);
    float *c = (float *)malloc(size);

    for (int i = 0; i < n; i++) {
        float af = rand() / double(RAND_MAX);
        float bf = rand() / double(RAND_MAX);
        a[i] = af;
        b[i] = bf;
    }

    // 定义空指针。
    float *da = NULL;
    float *db = NULL;
    float *dc = NULL;

    // 申请显存，da 指向申请的显存，注意 mcMalloc 函数传入指针的指针 (指向申请得到的显存的指针)。
    mcMalloc((void **)&da, size);
    mcMalloc((void **)&db, size);
    mcMalloc((void **)&dc, size);

    // 把内存的东西拷贝到显存，也就是把 a, b, c 里面的东西拷贝到 d_a, d_b, d_c 中。
    mcMemcpy(da,a,size,mcMemcpyHostToDevice);
    mcMemcpy(db,b,size,mcMemcpyHostToDevice);

    struct timeval t1, t2;

    // 计算线程块和网格的数量。
    int threadPerBlock_array[8]={1,16,32,64,128,256,512,1024};
    for(int i=0;i<8;i++){
        int threadPerBlock = threadPerBlock_array[i];
        int blockPerGrid = (n + threadPerBlock - 1)/threadPerBlock;
        printf("threadPerBlock: %d \nblockPerGrid: %d\n", threadPerBlock,blockPerGrid);

        gettimeofday(&t1, NULL);

        // 调用核函数。
        gpuVectorAddKernel<<< blockPerGrid, threadPerBlock >>> (da, db, dc, n);

        gettimeofday(&t2, NULL);

        mcMemcpy(c,dc,size,mcMemcpyDeviceToHost);

        // for (int i = 0; i < 10; i++) 
        //     cout<<vecA[i]<<" "<<vecB[i]<<" "<<vecC[i]<< endl;

        double timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
        cout << "threadPerBlock: " << threadPerBlock << "timeuse: " << timeuse << endl;

    }
    
    mcFree(da);
    mcFree(db);
    mcFree(dc);

    free(a);
    free(b);
    free(c);
    return 0;
}


```

#### 运行结果

<img src=".\3.1.png">

### Exercise 2

#### 参考代码

```c++
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <mc_runtime_api.h>

using namespace std;

// 要用 __global__ 来修饰。
// 输入指向3段显存的指针名。
__global__ void gpuVectorAddKernel(float* A_d,float* B_d,float* C_d, int N)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    // printf("threadIdx.x:%d  blockDim.x:%d  blockIdx.x:%d\n",threadIdx.x,blockDim.x,blockIdx.x);
    if (i < N) C_d[i] = A_d[i] + B_d[i];
}

int main(int argc, char *argv[]) {

    int n = 256;
    cout << n << endl;

    size_t size = n * sizeof(float);

    // host memory
    float *a = (float *)malloc(size);
    float *b = (float *)malloc(size);
    float *c = (float *)malloc(size);

    for (int i = 0; i < n; i++) {
        float af = rand() / double(RAND_MAX);
        float bf = rand() / double(RAND_MAX);
        a[i] = af;
        b[i] = bf;
    }

    // 定义空指针。
    float *da = NULL;
    float *db = NULL;
    float *dc = NULL;

    // 申请显存，da 指向申请的显存，注意 mcMalloc 函数传入指针的指针 (指向申请得到的显存的指针)。
    mcMalloc((void **)&da, size);
    mcMalloc((void **)&db, size);
    mcMalloc((void **)&dc, size);

    // 把内存的东西拷贝到显存，也就是把 a, b, c 里面的东西拷贝到 d_a, d_b, d_c 中。
    mcMemcpy(da,a,size,mcMemcpyHostToDevice);
    mcMemcpy(db,b,size,mcMemcpyHostToDevice);

    struct timeval t1, t2;

    // 计算线程块和网格的数量。
    int threadPerBlock_array[2]={1,256};
    for(int i=0;i<2;i++){
        int threadPerBlock = threadPerBlock_array[i];
        int blockPerGrid = (n + threadPerBlock - 1)/threadPerBlock;
        printf("threadPerBlock: %d \nblockPerGrid: %d\n", threadPerBlock,blockPerGrid);

        gettimeofday(&t1, NULL);

        // 调用核函数。
        gpuVectorAddKernel<<< blockPerGrid, threadPerBlock >>> (da, db, dc, n);

        gettimeofday(&t2, NULL);

        mcMemcpy(c,dc,size,mcMemcpyDeviceToHost);

        // for (int i = 0; i < 10; i++) 
        //     cout<<vecA[i]<<" "<<vecB[i]<<" "<<vecC[i]<< endl;

        double timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
        cout << "threadPerBlock: " << threadPerBlock << "timeuse: " << timeuse << endl;

    }
    
    mcFree(da);
    mcFree(db);
    mcFree(dc);

    free(a);
    free(b);
    free(c);
    return 0;
}

```

#### 运行结果

<img src=".\3.2.png">

### Exercise 3 

执行每个数值计算的速度并没有CPU快，CPU更适合处理逻辑控制密集的计算任务，GPU更适合处理数据密集的计算任务

### Exercise 4

#### 参考代码

```c
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <mc_runtime_api.h>

using namespace std;


__global__ void matrixMultiplication(int *A_d,int *B_d,int *Result_d,int width){
    int i=threadIdx.x+blockDim.x*blockIdx.x;
    int j=threadIdx.y+blockDim.y*blockIdx.y;
    int sum=0;
    int count;
    for(count=0;count<width;count++) sum+=A_d[j*width+count]*B_d[count*width+i];
    Result_d[j*width+i]=sum;
}

int main(){
    int *a,*b,*result;
    int col=15,row=15;
    // host memory
    a=(int *)malloc(sizeof(int)*row*col);
    b=(int *)malloc(sizeof(int)*row*col);
    result=(int *)malloc(sizeof(int)*row*col);
    // initialize
    int i;
    for(i=0;i<row*col;i++){
        a[i]=(int)rand() %15;
        b[i]=(int)rand() %15;
    }
    // 定义空指针
    int *da,*db,*d_result;
    // 申请显存
    mcMalloc((void **)&da,sizeof(int)*row*col);
    mcMalloc((void **)&db,sizeof(int)*row*col);
    mcMalloc((void **)&d_result,sizeof(int)*row*col);
    // 把内存的东西拷贝到显存
    mcMemcpy(da,a,sizeof(int)*row*col,mcMemcpyHostToDevice);
    mcMemcpy(db,b,sizeof(int)*row*col,mcMemcpyHostToDevice);
    // 计算线程块和网格的数量
    dim3 threadPerBlock(16,16);
    dim3 blockNumber((col+threadPerBlock.x-1)/ threadPerBlock.x, (row+threadPerBlock.y-1)/ threadPerBlock.y );
    // 调用kernel函数
    matrixMultiplication<<<blockNumber,threadPerBlock>>>(da,db,d_result,col);
    // 把显存的东西拷贝回内存
    mcMemcpy(result,d_result,sizeof(int)*row*col,mcMemcpyDeviceToHost);
    // print矩阵，这里row和col相等，所以统一用col表示
    int j;
    printf("a:\n");
    for(i=0;i<col;i++){
        for(j=0;j<col;j++){
            printf("%d ",a[i*col+j]);
        }
        printf("\n");
    }
    printf("b:\n");
    for(i=0;i<col;i++){
        for(j=0;j<col;j++){
            printf("%d ",b[i*col+j]);
        }
        printf("\n");
    }
    printf("result:\n");
    for(i=0;i<col;i++){
        for(j=0;j<col;j++){
            printf("%d ",result[i*col+j]);
        }
        printf("\n");
    }
    // free
    free(a);
    free(b);
    free(result);
    mcFree(da);
    mcFree(db);
    mcFree(d_result);
    return 0;
}
```

 

#### 运行结果

<img src=".\T4运行结果.png">

## Chapter 5

### 5.2.9

#### Exercise 1

##### 参考代码

```c
#include <iostream>
#include<mc_runtime_api.h>
#include <stdio.h>
using namespace std;


__global__ void print()
{
    printf("blockIdx.x:%d threadIdx.x:%d\n",blockIdx.x, threadIdx.x);
}

int main(void)
{
    const dim3 block_size(16);
    print<<<10, block_size>>>();
    mcDeviceSynchronize();
    return 0;
}


```

##### 运行结果（一部分）

<img src=".\5.2.9.1运行结果\1.png">

<img src=".\5.2.9.1运行结果\2.png">

<img src=".\5.2.9.1运行结果\3.png">

同一个wave内部thread的执行是顺序的。block的执行不是顺序的。

在MXMACA中，wave对程序员来说是透明的，它的大小可能会随着硬件的发展发生变化，在当前版本的MXMACA中，每个wave是由64个thread组成的。由64个thread组成的wave是MACA程序执行的最小单位，并且同一个wave是串行的。在一个SM中可能同时有来自不同block的wave。当一个block中的wave在进行访存或者同步等高延迟操作时，另一个block可以占用SM中的计算资源。这样，在SM内就实现了简单的乱序执行。不同block之间的执行没有顺序，完全并行。并且，一个sm只会执行一个block里的wave，当该block里的wave执行完才会执行其他block里的wave。

#### Exercise 2

##### 参考代码

```c
#include <iostream>
#include<mc_runtime_api.h>
#include <stdio.h>
using namespace std;


__global__ void print()
{
    printf("blockIdx.x:%d threadIdx.x:%d threadIdx.y:%d threadIdx.z:%d\n",blockIdx.x, threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(void)
{
    const dim3 block_size(16);
    print<<<10, block_size>>>();
    mcDeviceSynchronize();
    return 0;
}


```



##### 运行结果

<img src=".\5.2.9.2运行结果\1.png">

<img src=".\5.2.9.2运行结果\2.png">

<img src=".\5.2.9.2运行结果\3.png">

没有定义，默认为0.

可以在定义block_size时对三个维度的size都进行设置（注意三者的乘积不可以超过maxThreadsPerBlock）。

### 5.4.4(待更正)

#### Exercise 1

##### 参考代码

```c
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

```

`mandelbrot` 函数计算给定复数的Mandelbrot迭代次数。

`mandelbrot_kernel` 核函数在每个线程中计算像素点的迭代次数，并将结果存储在 `output` 数组中。

`mandelbrotSet` 函数用于分配和释放设备内存，并在设备上调用核函数进行计算。

最后，在 `main` 函数中，我们定义了图像的宽度、高度和最大迭代次数，并调用 `mandelbrotSet` 函数来计算Mandelbrot集合，并将结果存储在 `output` 数组中。

可以根据需要修改图像的宽度、高度、迭代次数以及计算范围，然后将结果输出或保存为图像文件。

## Chapter 6

### Exercise 1

改用统一寻址方式简化vectorAdd

#### 参考代码

```c++
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <mc_runtime_api.h>

using namespace std;

__global__ void vectorAdd(float* A_d, float* B_d, float* C_d, int N){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) C_d[i] = A_d[i] + B_d[i] + 0.0f;
}

int main(int argc,char *argv[]){
    int n = atoi(argv[1]);
    cout << n << endl;
    
    float *A,*B,*C;
    mcMallocManaged(&A,n*sizeof(float));
    mcMallocManaged(&B,n*sizeof(float));
    mcMallocManaged(&C,n*sizeof(float));

    for(int i=0;i<n;i++){
        A[i]=rand() / double(RAND_MAX);
        B[i]=rand() / double(RAND_MAX);
    }
    int threadPerBlock=256;
    int blockPerGrid=(n + threadPerBlock - 1)/threadPerBlock;
    printf("threadPerBlock: %d \nblockPerGrid: %d \n",threadPerBlock,blockPerGrid);
    vectorAdd<<<blockPerGrid,threadPerBlock>>>(A,B,C,n);
    mcDeviceSynchronize();
    for(int i=0;i<n;i++){
        printf("A[%d]:%f  B[%d]:%f  C[%d]:%f\n",i,A[i],i,B[i],i,C[i]);
    }
    mcFree(A);
    mcFree(B);
    mcFree(C);
    return 0;
}
```

#### 运行结果

文件另存为vectorAdd.cpp

在控制台输入：

```
mxcc -x maca vectorAdd.cpp -o vectorAdd

./vectorAdd 10 （可以自定义其他数值）
```

<img src=".\统一内存寻址运行结果.png">

### Exercise 2

```c++
#include <stdio.h>
#include <mc_runtime_api.h>
#include <math.h>
#include <mc_common.h>
#include <sys/time.h>
#include <iostream>
using namespace std;

#define M 512
#define K 512
#define N 512

void initial(float *array, int size)
{
	for (int i = 0; i < size; i++)
	{
		array[i] = (float)(rand() % 10 + 1);
	}
}

//核函数（静态共享内存版）
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns)
{
	//分配共享内存
	// __shared__ float sharedM[blockDim.y][blockDim.x];
	// __shared__ float sharedN[blockDim.x][blockDim.y];
	__shared__ float sharedM[16][32];
	__shared__ float sharedN[16][32];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;

	float Csub = 0.0;
	
	//将保存在全局内存中的矩阵M&N分块存放到共享内存中
	for (int i = 0; i < (int)(ceil((float)numAColumns / blockDim.x)); i++)
	{
		if (i*blockDim.x + tx < numAColumns && row < numARows)
			sharedM[ty][tx] = A[row*numAColumns + i * blockDim.x + tx];
		else
			sharedM[ty][tx] = 0.0;

		if (i*blockDim.y + ty < numBRows && col < numBColumns)//分割N矩阵
			sharedN[ty][tx] = B[(i*blockDim.y + ty)*numBColumns + col];
		else
			sharedN[ty][tx] = 0.0;
		__syncthreads();

		for (int j = 0; j < blockDim.x; j++)//分块后的矩阵相乘
			Csub += sharedM[ty][j] * sharedN[j][tx];
		__syncthreads();
	}

	if (row < numCRows && col < numCColumns)//将计算后的矩阵块放到结果矩阵C中
		C[row*numCColumns + col] = Csub;
}


int main(int argc, char **argv)
{
	int Axy = M * K;
	int Bxy = K * N;
	int Cxy = M * N;

	float *h_A, *h_B, *h_C;
	h_A = (float*)malloc(Axy * sizeof(float));
	h_B = (float*)malloc(Bxy * sizeof(float));

	h_C = (float*)malloc(Cxy * sizeof(float));

	initial(h_A, Axy);
	initial(h_B, Bxy);
	
	float *d_A, *d_B, *d_C;
	mcMalloc((void**)&d_A, Axy * sizeof(float));
	mcMalloc((void**)&d_B, Bxy * sizeof(float));
	mcMalloc((void**)&d_C, Cxy * sizeof(float));

	mcMemcpy(d_A, h_A, Axy * sizeof(float), mcMemcpyHostToDevice);
	mcMemcpy(d_B, h_B, Bxy * sizeof(float), mcMemcpyHostToDevice);
	
    int dimx = 32;
    int dimy = 16;
    dim3 block(dimx, dimy);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
	struct timeval t1, t2;
    gettimeofday(&t1, NULL);
	matrixMultiplyShared <<< grid, block >>> (d_A, d_B, d_C, M, K, K, N, M, N);
	mcMemcpy(h_C, d_C, Cxy * sizeof(float), mcMemcpyDeviceToHost);
	gettimeofday(&t2, NULL);
    double timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    cout << "timeuse: " << timeuse << endl;
    mcFree(d_A);
    mcFree(d_B);
    mcFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
}

```



