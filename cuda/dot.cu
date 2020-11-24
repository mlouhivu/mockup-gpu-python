#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dot_(int n, double *x, double *y, double *z)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    extern __shared__ double buffer[];
    int i;

    double sum = 0.0f;
    for (i = tid; i < n; i += stride) {
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
        *z = buffer[0];
}

int main(void)
{
    int i;
    const int n = 10000;
    double x[n], y[n];
    double z, z_ref;
    double *x_, *y_, *z_;

    // initialise data and calculate reference values on CPU
    z_ref = 0.0f;
    for (i=0; i < n; i++) {
        x[i] = sin(i) * 2.3;
        y[i] = cos(i) * 1.1;
        z_ref += x[i] * y[i];
    }

    // allocate + copy initial values
    cudaMalloc((void **) &x_, sizeof(double) * n);
    cudaMalloc((void **) &y_, sizeof(double) * n);
    cudaMalloc((void **) &z_, sizeof(double));
    cudaMemcpy(x_, x, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * n, cudaMemcpyHostToDevice);

    // calculate axpy on GPU
    dot_<<<1,32*256>>>(n, x_, y_, z_);

    // copy result back to host and print with reference
    cudaMemcpy(&z, z_, sizeof(double), cudaMemcpyDeviceToHost);
    printf("reference: %f\n   result: %f\n", z_ref, z);

    return 0;
}
