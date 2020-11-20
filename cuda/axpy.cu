#include <stdio.h>
#include <cuda_runtime.h>

__global__ void daxpy_(int n, double a, double *x, double *y)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        y[tid] += a * x[tid];
    }
}

__global__ void daxpy_mono_(int n, double a, double *x, double *y)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
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
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[n-2], y[n-1]);

    // allocate + copy initial values
    cudaMalloc((void **) &x_, sizeof(double) * n);
    cudaMalloc((void **) &y_, sizeof(double) * n);
    cudaMemcpy(x_, x, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * n, cudaMemcpyHostToDevice);

    // save initial values for a retry
    memcpy(x, y, sizeof(double) * n);

    // calculate axpy on GPU
    daxpy_<<<32,256>>>(n, a, x_, y_);

    // copy result back to host and print with reference
    cudaMemcpy(&y, y_, sizeof(double) * n, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[n-2], y_ref[n-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[n-2], y[n-1]);

    /* MONOLITHIC kernel */
    // copy initial values back for a retry
    cudaMemcpy(y_, x, sizeof(double) * n, cudaMemcpyHostToDevice);

    // calculate gridsize for a one-pass kernel launch
    int blockSize = 256;
    int gridSize = ((int) (n / (blockSize * 32)) + 1) * 32;
    printf("<<<gridSize,blockSize>>> = <<<%d,%d>>>\n", gridSize, blockSize);

    // calculate axpy on GPU using a monolithic kernel
    daxpy_mono_<<<gridSize,blockSize>>>(n, a, x_, y_);

    // copy result back to host and print with reference
    cudaMemcpy(&y, y_, sizeof(double) * n, cudaMemcpyDeviceToHost);
    printf("   reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[n-2], y_ref[n-1]);
    printf("result(mono): %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[n-2], y[n-1]);

    return 0;
}
