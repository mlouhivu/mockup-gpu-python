#include <hip/hip_runtime.h>
#include <stdio.h>

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
    hipMalloc((void **) &x_, sizeof(double) * n);
    hipMalloc((void **) &z_, sizeof(double));
    hipMalloc((void **) &buffer_, sizeof(double) * blocks);
    hipMemcpy(x_, x, sizeof(double) * n, hipMemcpyHostToDevice);

    // calculate dot product on GPU (partial sums)
    hipLaunchKernelGGL(sum_, blocks, threads, threads * sizeof(double), 0,
                       n, x_, buffer_);
    // reduce partial sums
    hipLaunchKernelGGL(sum_, 1, blocks, blocks * sizeof(blocks), 0,
                       blocks, buffer_, z_);

    // copy result back to host and print with reference
    hipMemcpy(&z, z_, sizeof(double), hipMemcpyDeviceToHost);
    printf(" reference: %f\n    result: %f\n", z_ref, z);

    return 0;
}
