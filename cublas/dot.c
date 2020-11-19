#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(void)
{
    int i;
    const int n = 100;
    double x[n], y[n];
    double *x_, *y_, *z_;
    double z_ref = 0.0;
    double z;
    cublasHandle_t handle;

    // initialise data and calculate reference value on CPU
    for (i=0; i < n; i++) {
        x[i] = sin(i) * 2.3;
        y[i] = cos(i) * 1.1;
        z_ref += x[i] * y[i];
    }
    cublasCreate(&handle);

    // use GPU pointers to keep data on GPU
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    // allocate + copy initial values
    cudaMalloc((void **)(&x_), sizeof(double) * n);
    cudaMalloc((void **)(&y_), sizeof(double) * n);
    cudaMalloc((void **)(&z_), sizeof(double));
    cudaMemcpy(x_, x, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * n, cudaMemcpyHostToDevice);

    // calculate dot on GPU
    cublasDdot(handle, n, x_, 1, y_, 1, z_);

    // copy result back to host and print with reference
    cudaMemcpy(&z, z_, sizeof(double), cudaMemcpyDeviceToHost);
    printf("reference: %f\n   result: %f\n", z_ref, z);

    cublasDestroy(handle);
    return 0;
}
