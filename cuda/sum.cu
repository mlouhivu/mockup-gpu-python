#include <stdio.h>
#include <cuda_runtime.h>

__global__ void sum_(int n, double *x, double *y)
{
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    extern __shared__ double buffer[];
    int i;

    double sum = 0.0f;
    for (i = gid; i < n; i += stride) {
        sum += x[i];
    }
    buffer[tid] = sum;
    __syncthreads();

    i = blockDim.x / 2;
    while (i > 0) {
        if (tid < i)
            buffer[tid] += buffer[tid + i];
        i = i / 2;
        __syncthreads();
    }

    if (tid == 0)
        y[blockIdx.x] = buffer[0];
}

int main(void)
{
    const int blocks = 32;
    const int threads = 256;
    const int n = 10000;
    double x[n];
    double z, z_ref;
    double *x_, *z_, *buffer_;
    int i;

    // initialise data and calculate reference values on CPU
    z_ref = 0.0f;
    for (i=0; i < n; i++) {
        x[i] = sin(i) * 2.3;
        z_ref += x[i];
    }

    // allocate + copy initial values
    cudaMalloc((void **) &x_, sizeof(double) * n);
    cudaMalloc((void **) &z_, sizeof(double));
    cudaMalloc((void **) &buffer_, sizeof(double) * blocks);
    cudaMemcpy(x_, x, sizeof(double) * n, cudaMemcpyHostToDevice);

    // calculate partial sums on GPU
    sum_<<<blocks, threads, threads * sizeof(double)>>>(n, x_, buffer_);
    // reduce partial sums
    sum_<<<1, blocks, blocks * sizeof(double)>>>(blocks, buffer_, z_);

    // copy result back to host and print with reference
    cudaMemcpy(z, z_, sizeof(double), cudaMemcpyDeviceToHost);
    printf(" reference: %f\n    result: %f\n", z_ref, z);

    return 0;
}
