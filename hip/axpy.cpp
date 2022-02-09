#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void daxpy_(int n, double a, double *x, double *y)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        y[tid] += a * x[tid];
    }
}

__global__ void saxpy_(int n, float a, float *x, float *y)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        y[tid] += a * x[tid];
    }
}

int main(void)
{
    int i;
    const int n = 10000;
    double a = 3.4;
    double x[n], y[n], y_ref[n];
    double *x_, *y_;

    // initialise data and calculate reference values on CPU
    for (i=0; i < n; i++) {
        x[i] = sin(i) * 2.3;
        y[i] = cos(i) * 1.1;
        y_ref[i] = a * x[i] + y[i];
    }

    // allocate + copy initial values
    hipMalloc((void **) &x_, sizeof(double) * n);
    hipMalloc((void **) &y_, sizeof(double) * n);
    hipMemcpy(x_, x, sizeof(double) * n, hipMemcpyHostToDevice);
    hipMemcpy(y_, y, sizeof(double) * n, hipMemcpyHostToDevice);

    // calculate axpy on GPU
    dim3 blocks(32);
    dim3 threads(256);
    hipLaunchKernelGGL(daxpy_, blocks, threads, 0, 0, n, a, x_, y_);

    // copy result back to host and print with reference
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[n-2], y[n-1]);
    hipMemcpy(y, y_, sizeof(double) * n, hipMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[n-2], y_ref[n-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[n-2], y[n-1]);

    return 0;
}
