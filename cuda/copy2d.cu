#include <stdio.h>
#include <cuda_runtime.h>

__global__ void copy2d_(double *src, double *tgt,
                        int ni, int nj,
                        int mi, int mj,
                        int oi, int oj)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int stridex = gridDim.x * blockDim.x;
    int stridey = gridDim.y * blockDim.y;
    int t, s;
    int i, j;

    for (i = tidy; i < mi; i += stridey) {
        t = mj * i;
        s = nj * (i + oi) + oj;
        for (j = tidx; j < mj; j += stridex) {
            tgt[j + t] = src[j + s];
        }
    }
}

__global__ void copy2d_fortran_(double *src, double *tgt,
                                int ni, int nj,
                                int mi, int mj,
                                int oi, int oj)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int stridex = gridDim.x * blockDim.x;
    int stridey = gridDim.y * blockDim.y;
    int t, s;
    int i, j;

    for (i = tidy; i < mj; i += stridey) {
        t = mi * i;
        s = ni * (i + oj) + oi;
        for (j = tidx; j < mi; j += stridex) {
            tgt[j + t] = src[j + s];
        }
    }
}


int main(void)
{
    int i, j, s, t;
    int dimx[2] = {100,100};
    int dimy[2] = {10,10};
    const int n = dimx[0] * dimx[1];
    const int m = dimy[0] * dimy[1];
    int position[2] = {22,44};
    double x[n], y[m], y_ref[m];
    double *x_, *y_;

    dim3 blocks(32, 32, 1);
    dim3 threads(32, 8, 1);

    // initialise data
    for (i=0; i < n; i++) {
        x[i] = (double) i / 1000.0;
    }
    for (i=0; i < m; i++) {
        y[i] = 0.0;
    }
    // copy reference values (C ordered)
    for (i=0; i < dimy[0]; i++) {
        for (j=0; j < dimy[1]; j++) {
            t = dimy[1] * i + j;
            s = dimx[1] * (i + position[0]) + j + position[1];
            y_ref[t] = x[s];
        }
    }

    // allocate + copy initial values
    cudaMalloc((void **) &x_, sizeof(double) * n);
    cudaMalloc((void **) &y_, sizeof(double) * m);
    cudaMemcpy(x_, x, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m, cudaMemcpyHostToDevice);

    // copy subarray on GPU
    copy2d_<<<blocks, threads>>>(x_, y_, dimx[0], dimx[1], dimy[0], dimy[1],
                                 position[0], position[1]);

    // copy results back to host and print with reference
    printf("\nC ordered\n---------\n");
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(y, y_, sizeof(double) * m, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);

    // copy reference values (Fortran ordered)
    for (i=0; i < dimy[1]; i++) {
        for (j=0; j < dimy[0]; j++) {
            t = dimy[0] * i + j;
            s = dimx[0] * (i + position[1]) + j + position[0];
            y_ref[t] = x[s];
        }
    }

    // copy subarray on GPU (Fortran ordered)
    copy2d_fortran_<<<blocks, threads>>>(x_, y_,
                                         dimx[0], dimx[1], dimy[0], dimy[1],
                                         position[0], position[1]);

    // copy results back to host and print with reference
    printf("\nFortran ordered\n---------\n");
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(y, y_, sizeof(double) * m, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);

    return 0;
}
