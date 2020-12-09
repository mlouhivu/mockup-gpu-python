#include <stdio.h>
#include <cuda_runtime.h>

__global__ void paste_(double *src, double *tgt, int n, int m, int offset)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        if (tid + offset < m)
            tgt[tid + offset] = src[tid];
    }
}


int main(void)
{
    int i;
    const int n = 1000;
    const int m = 10000;
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
    for (i=0; i < n; i++) {
        y_ref[i + position] = x[i];
    }

    // allocate + copy initial values
    cudaMalloc((void **) &x_, sizeof(double) * n);
    cudaMalloc((void **) &y_, sizeof(double) * m);
    cudaMemcpy(x_, x, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m, cudaMemcpyHostToDevice);

    // copy into subarray on GPU
    paste_<<<blocks, threads>>>(x_, y_, n, m, position);

    // copy result back to host and print with reference
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[position], y[position+1], y[position+2], y[position+3],
            y[position+n-2], y[position+n-1]);
    cudaMemcpy(&y, y_, sizeof(double) * m, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[position], y_ref[position+1], y_ref[position+2],
            y_ref[position+3], y_ref[position+n-2], y_ref[position+n-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[position], y[position+1], y[position+2], y[position+3],
            y[position+n-2], y[position+n-1]);

    return 0;
}
