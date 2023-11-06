#include <iostream>
#include <cstdlib>
#include <sys/time.h>

using namespace std;

void cpuVectorAdd(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char *argv[]) {

    int n = atoi(argv[1]);
    cout << n << endl;

    size_t size = n * sizeof(float);

    // host memery
    float *a = (float *)malloc(size); //分配一段内存，使用指针 a 指向它。
    float *b = (float *)malloc(size);
    float *c = (float *)malloc(size);

    // for 循环产生一些随机数，并放在分配的内存里面。
    for (int i = 0; i < n; i++) {
        float af = rand() / double(RAND_MAX);
        float bf = rand() / double(RAND_MAX);
        a[i] = af;
        b[i] = bf;
    }

    struct timeval t1, t2;

    // gettimeofday 函数来得到精确时间。它的精度可以达到微秒，是C标准库的函数。
    gettimeofday(&t1, NULL);

    // 输入指向3段内存的指针名，也就是 a, b, c。
    cpuVectorAdd(a, b, c, n);

    gettimeofday(&t2, NULL);

    //for (int i = 0; i < 10; i++) 
    //    cout << vecA[i] << " " << vecB[i] << " " << vecC[i] << endl;
    double timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    cout << timeuse << endl;

    // free 函数把申请的3段内存释放掉。
    free(a);
    free(b);
    free(c);
    return 0;
}
