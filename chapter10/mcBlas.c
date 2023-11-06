#include <stdio.h>  
#include <stdlib.h>  
#include <string.h>  
#include <mc_runtime_api.h>  
#include "mcblas.h"
  
/* cpu implementation of sgemm */  
static void cpu_sgemm(int m, int n, int k, float alpha, const float *A, const float *B, float beta, float *C_in,  
                      float *C_out) {  
  int i;  
  int j;  
  int kk;  
  
  for (i = 0; i < m; ++i) {  
    for (j = 0; j < n; ++j) {  
      float prod = 0;  
  
      for (kk = 0; kk < k; ++kk) {  
        prod += A[kk * m + i] * B[j * k + kk];  
      }  
  
      C_out[j * m + i] = alpha * prod + beta * C_in[j * m + i];  
    }  
  }  
}  
  
int main(int argc, char **argv) {  
  float *h_A;  
  float *h_B;  
  float *h_C;  
  float *h_C_ref;  
  float *d_A = 0;  
  float *d_B = 0;  
  float *d_C = 0;  
  float alpha = 1.0f;  
  float beta = 0.0f;  
  int m = 256;  
  int n = 128;  
  int k = 64;  
  int size_a = m * n;  // the element num of A matrix  
  int size_b = n * k;  // the element num of B matrix  
  int size_c = m * n;  // the element num of C matrix  
  float error_norm;  
  float ref_norm;  
  float diff;  
  mcblasHandle_t handle;  
  mcblasStatus_t status;  
  
  /* Initialize mcBLAS */  
  status = mcblasCreate(&handle);  
  if (status != MCBLAS_STATUS_SUCCESS) {  
    fprintf(stderr, "Init failed\n");  
    return EXIT_FAILURE;  
  }  
  
  /* Allocate host memory for A/B/C matrix*/  
  h_A = (float *)malloc(size_a * sizeof(float));  
  if (h_A == NULL) {  
    fprintf(stderr, "A host memory allocation failed\n");  
    return EXIT_FAILURE;  
  }  
  h_B = (float *)malloc(size_b * sizeof(float));  
  if (h_B == NULL) {  
    fprintf(stderr, "B host memory allocation failed\n");  
    return EXIT_FAILURE;  
  }  
  h_C = (float *)malloc(size_c * sizeof(float));  
  if (h_C == 0) {  
    fprintf(stderr, "C host memory allocation failed\n");  
    return EXIT_FAILURE;  
  }  
  h_C_ref = (float *)malloc(size_c * sizeof(float));  
  if (h_C_ref == 0) {  
    fprintf(stderr, "C_ref host memory allocation failed\n");  
    return EXIT_FAILURE;  
  }  
  
  /* Fill the matrices with test data */  
  for (int i = 0; i < size_a; ++i) {  
    h_A[i] = cos(i + 0.125);  
  }  
  for (int i = 0; i < size_b; ++i) {  
    h_B[i] = cos(i - 0.125);  
  }  
  for (int i = 0; i < size_c; ++i) {  
    h_C[i] = sin(i + 0.25);  
  }  
  
  /* Allocate device memory for the matrices */  
  if (mcMalloc((void **)(&d_A), size_a * sizeof(float)) != mcSuccess) {  
    fprintf(stderr, "A device memory allocation failed\n");  
    return EXIT_FAILURE;  
  }  
  if (mcMalloc((void **)(&d_B), size_b * sizeof(float)) != mcSuccess) {  
    fprintf(stderr, "B device memory allocation failed\n");  
    return EXIT_FAILURE;  
  }  
  if (mcMalloc((void **)(&d_C), size_c * sizeof(float)) != mcSuccess) {  
    fprintf(stderr, "C device memory allocation failed\n");  
    return EXIT_FAILURE;  
  }  
  
  /* Initialize the device matrices with the host matrices */  
  if (mcblasSetVector(size_a, sizeof(float), h_A, 1, d_A, 1) != MCBLAS_STATUS_SUCCESS) {  
    fprintf(stderr, "Copy A from host to device failed\n");  
    return EXIT_FAILURE;  
  }  
  if (mcblasSetVector(size_b, sizeof(float), h_B, 1, d_B, 1) != MCBLAS_STATUS_SUCCESS) {  
    fprintf(stderr, "Copy B from host to device failed\n");  
    return EXIT_FAILURE;  
  }  
  if (mcblasSetVector(size_c, sizeof(float), h_C, 1, d_C, 1) != MCBLAS_STATUS_SUCCESS) {  
    fprintf(stderr, "Copy C from host to device failed\n");  
    return EXIT_FAILURE;  
  }  
  
  /* compute the reference result */  
  cpu_sgemm(m, n, k, alpha, h_A, h_B, beta, h_C, h_C_ref);  
  
  /* Performs operation using mcblas */  
  status = mcblasSgemm(handle, MCBLAS_OP_N, MCBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, n, &beta, d_C, k);  
  if (status != MCBLAS_STATUS_SUCCESS) {  
    fprintf(stderr, "Sgemm kernel execution failed\n");  
    return EXIT_FAILURE;  
  }  
  /* Read the result back */  
  status = mcblasGetVector(size_c, sizeof(float), d_C, 1, h_C, 1);  
  if (status != MCBLAS_STATUS_SUCCESS) {  
    fprintf(stderr, "C data reading failed\n");  
    return EXIT_FAILURE;  
  }  
  
  /* Check result against reference */  
  error_norm = 0;  
  ref_norm = 0;  
  
  for (int i = 0; i < size_c; ++i) {  
    diff = h_C_ref[i] - h_C[i];  
    error_norm += diff * diff;  
    ref_norm += h_C_ref[i] * h_C_ref[i];  
  }  
  
  error_norm = (float)sqrt((double)error_norm);  
  ref_norm = (float)sqrt((double)ref_norm);  
  
  if (error_norm / ref_norm < 1e-6f) {  
    printf("McBLAS test passed.\n");  
  } else {  
    printf("McBLAS test failed.\n");  
  }  
  
  /* Memory clean up */  
  free(h_A);  
  free(h_B);  
  free(h_C);  
  free(h_C_ref);  
  
  if (mcFree(d_A) != mcSuccess) {  
    fprintf(stderr, "A device mem free failed\n");  
    return EXIT_FAILURE;  
  }  
  
  if (mcFree(d_B) != mcSuccess) {  
    fprintf(stderr, "B device mem free failed\n");  
    return EXIT_FAILURE;  
  }  
  
  if (mcFree(d_C) != mcSuccess) {  
    fprintf(stderr, "C device mem free failed\n");  
    return EXIT_FAILURE;  
  }  
  
  /* Shutdown */  
  status = mcblasDestroy(handle);  
  if (status != MCBLAS_STATUS_SUCCESS) {  
    fprintf(stderr, "Destory failed\n");  
    return EXIT_FAILURE;  
  }  
  
  return EXIT_SUCCESS;  
}
