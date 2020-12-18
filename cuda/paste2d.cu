#include <stdio.h>
#include <cuda_runtime.h>

__global__ void paste2d_(double *src, double *tgt,
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

    for (i = tidy; i < ni; i += stridey) {
        s = nj * i;
        t = mj * (i + oi) + oj;
        for (j = tidx; j < nj; j += stridex) {
            tgt[j + t] = src[j + s];
        }
    }
}

__global__ void paste2d_fortran_(double *src, double *tgt,
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

    for (i = tidy; i < nj; i += stridey) {
        s = ni * i;
        t = mi * (i + oj) + oi;
        for (j = tidx; j < ni; j += stridex) {
            tgt[j + t] = src[j + s];
        }
    }
}


int main(void)
{
    int i, j, s, t;
    int dimx[2] = {10,10};
    int dimy[2] = {100,100};
    const int n = dimx[0] * dimx[1];
    const int m = dimy[0] * dimy[1];
    int position[2] = {22,44};
    int start, end;
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
    for (i=0; i < dimx[0]; i++) {
        for (j=0; j < dimx[1]; j++) {
            s = dimx[1] * i + j;
            t = dimy[1] * (i + position[0]) + j + position[1];
            y_ref[t] = x[s];
        }
    }

    // allocate + copy initial values
    cudaMalloc((void **) &x_, sizeof(double) * n);
    cudaMalloc((void **) &y_, sizeof(double) * m);
    cudaMemcpy(x_, x, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m, cudaMemcpyHostToDevice);

    // copy into subarray on GPU
    paste2d_<<<blocks, threads>>>(x_, y_, dimx[0], dimx[1], dimy[0], dimy[1],
                                  position[0], position[1]);

    // copy results back to host and print with reference
    printf("\nC ordered\n---------\n");
    start = dimy[1] * position[0] + position[1];
    end = dimy[1] * (position[0] + dimx[0] - 1) + position[1] + dimx[1] - 1;
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[start], y[start+1], y[start+2], y[start+3],
            y[end-1], y[end]);
    cudaMemcpy(&y, y_, sizeof(double) * m, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[start], y_ref[start+1], y_ref[start+2],
            y_ref[start+3], y_ref[end-1], y_ref[end]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[start], y[start+1], y[start+2], y[start+3],
            y[end-1], y[end]);

    // copy reference values (Fortran ordered)
    for (i=0; i < dimx[1]; i++) {
        for (j=0; j < dimx[0]; j++) {
            s = dimx[0] * i + j;
            t = dimy[0] * (i + position[1]) + j + position[0];
            y_ref[t] = x[s];
        }
    }

    // copy subarray on GPU (Fortran ordered)
    paste2d_fortran_<<<blocks, threads>>>(x_, y_,
                                          dimx[0], dimx[1], dimy[0], dimy[1],
                                          position[0], position[1]);

    // copy results back to host and print with reference
    printf("\nFortran ordered\n---------\n");
    start = dimy[0] * position[1] + position[0];
    end = dimy[0] * (position[1] + dimx[1] - 1) + position[0] + dimx[0] - 1;
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[start], y[start+1], y[start+2], y[start+3],
            y[end-1], y[end]);
    cudaMemcpy(&y, y_, sizeof(double) * m, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[start], y_ref[start+1], y_ref[start+2],
            y_ref[start+3], y_ref[end-1], y_ref[end]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[start], y[start+1], y[start+2], y[start+3],
            y[end-1], y[end]);

    return 0;
}
