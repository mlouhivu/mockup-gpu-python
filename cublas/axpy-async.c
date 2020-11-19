#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(void)
{
    int i;
    const int n = 100;
    double a = 3.4;
    double *a_;
    double x[n], y[n], y_ref[n];
    double *x_, *y_;
    cublasHandle_t handle;

    // initialise data and calculate reference values on CPU
    for (i=0; i < n; i++) {
        x[i] = sin(i) * 2.3;
        y[i] = cos(i) * 1.1;
        y_ref[i] = a * x[i] + y[i];
    }
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[n-2], y[n-1]);
    cublasCreate(&handle);

    // use GPU pointers to keep data on GPU
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    // allocate + copy initial values
    cudaMalloc((void **) &x_, sizeof(double) * n);
    cudaMalloc((void **) &y_, sizeof(double) * n);
    cudaMalloc((void **) &a_, sizeof(double));
    cudaMemcpy(x_, x, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(a_, &a, sizeof(double), cudaMemcpyHostToDevice);

    // calculate axpy on GPU
    cublasDaxpy(handle, n, a_, x_, 1, y_, 1);

    // copy result back to host and print with reference
    cudaMemcpy(&y, y_, sizeof(double) * n, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[n-2], y_ref[n-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[n-2], y[n-1]);

    cublasDestroy(handle);
    return 0;
}
