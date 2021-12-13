#include <stdio.h>
#include <cuda_runtime.h>

__global__ void copy_(double *src, double *tgt, int n, int m, int offset)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; tid < m; tid += stride) {
        if (tid + offset < n)
            tgt[tid] = src[tid + offset];
    }
}


int main(void)
{
    int i;
    const int n = 10000;
    const int m = 1000;
    int position = 22;
    double x[n], y[m], y_ref[m];
    double *x_, *y_;

    dim3 blocks(32, 1, 1);
    dim3 threads(256, 1, 1);

    // initialise data
    for (i=0; i < n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < m; i++) {
        y[i] = 0.0;
    }
    // copy reference values
    for (i=0; i < m; i++) {
        y_ref[i] = x[i + position];
    }

    // allocate + copy initial values
    cudaMalloc((void **) &x_, sizeof(double) * n);
    cudaMalloc((void **) &y_, sizeof(double) * m);
    cudaMemcpy(x_, x, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m, cudaMemcpyHostToDevice);

    // copy subarray on GPU
    copy_<<<blocks, threads>>>(x_, y_, n, m, position);

    // copy result back to host and print with reference
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(y, y_, sizeof(double) * m, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);

    return 0;
}
