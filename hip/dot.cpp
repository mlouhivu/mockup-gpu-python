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

__global__ void dot_(int n, double *x, double *y, double *z)
{
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    extern __shared__ double buffer[];
    int i;

    double sum = 0.0f;
    for (i = gid; i < n; i += stride) {
        sum += x[i] * y[i];
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
        z[blockIdx.x] = buffer[0];
}

void simulate_dot(int n, double *x, double *y, double *z,
                  int blockid, int blockdim, int griddim)
{
    int tid;
    int stride = griddim * blockdim;
    double buffer[blockdim];
    int i;

    for (tid=0; tid < blockdim; tid++) {
        int gid = tid + blockid * blockdim;
        buffer[tid] = 0.0f;
        for (i = gid; i < n; i += stride) {
            buffer[tid] += x[i] * y[i];
        }
    }

    i = blockdim / 2;
    while (i > 0) {
        for (tid=0; tid < blockdim; tid++) {
            if (tid < i)
                buffer[tid] += buffer[tid + i];
        }
        i = i / 2;
    }

    z[blockid] = buffer[0];
}

int main(void)
{
    const int blocks = 32;
    const int threads = 256;
    const int n = 10000;
    double x[n], y[n];
    double z, z_ref, z_sim;
    double *x_, *y_, *z_, *buffer_;
    double buffer[blocks];
    int i;

    // initialise data and calculate reference values on CPU
    z_ref = 0.0f;
    for (i=0; i < n; i++) {
        x[i] = sin(i) * 2.3;
        y[i] = cos(i) * 1.1;
        z_ref += x[i] * y[i];
    }
    // simulate GPU algorithm
    for (i=0; i < blocks; i++) {
        simulate_dot(n, x, y, buffer, i, 256, blocks);
    }
    z_sim = 0.0f;
    for (i=0; i < blocks; i++) {
        z_sim += buffer[i];
    }
    printf("simulation: %f\n", z_sim);

    // allocate + copy initial values
    hipMalloc((void **) &x_, sizeof(double) * n);
    hipMalloc((void **) &y_, sizeof(double) * n);
    hipMalloc((void **) &z_, sizeof(double));
    hipMalloc((void **) &buffer_, sizeof(double) * blocks);
    hipMemcpy(x_, x, sizeof(double) * n, hipMemcpyHostToDevice);
    hipMemcpy(y_, y, sizeof(double) * n, hipMemcpyHostToDevice);

    // calculate dot product on GPU (partial sums)
    hipLaunchKernelGGL(dot_, blocks, threads, threads * sizeof(double), 0,
                       n, x_, y_, buffer_);
    // reduce partial sums
    hipLaunchKernelGGL(sum_, 1, blocks, blocks * sizeof(blocks), 0,
                       blocks, buffer_, z_);

    // copy result back to host and print with reference
    hipMemcpy(&z, z_, sizeof(double), hipMemcpyDeviceToHost);
    printf(" reference: %f\n    result: %f\n", z_ref, z);

    return 0;
}
