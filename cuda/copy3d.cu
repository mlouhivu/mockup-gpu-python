#include <stdio.h>
#include <cuda_runtime.h>

__global__ void copy3d_(double *src, double *tgt,
                        int ni, int nj, int nk,
                        int mi, int mj, int mk,
                        int oi, int oj, int ok)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int tidz = threadIdx.z + blockIdx.z * blockDim.z;
    int stridex = gridDim.x * blockDim.x;
    int stridey = gridDim.y * blockDim.y;
    int stridez = gridDim.z * blockDim.z;
    int t, s, tz, sz;
    int i, j, k;

    for (i = tidz; i < mi; i += stridez) {
        tz = mk * mj * i;
        sz = nk * nj * (i + oi) + ok;
        for (j = tidy; j < mj; j += stridey) {
            t = tz + mk * j;
            s = sz + nk * (j + oj);
            for (k = tidx; k < mk; k += stridex) {
                /*
                t = mk * mj * i;
                  + mk * j
                  + k
                s = nk * nj * (i + oi)
                  + nk * (j + oj)
                  + k + ok
                tgt[t] = src[s];
                */
                tgt[k + t] = src[k + s];
            }
        }
    }
}


__global__ void copy3d_fortran_(double *src, double *tgt,
                                int ni, int nj, int nk,
                                int mi, int mj, int mk,
                                int oi, int oj, int ok)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int tidz = threadIdx.z + blockIdx.z * blockDim.z;
    int stridex = gridDim.x * blockDim.x;
    int stridey = gridDim.y * blockDim.y;
    int stridez = gridDim.z * blockDim.z;
    int t, s, tz, sz;
    int i, j, k;

    for (i = tidz; i < mk; i += stridez) {
        tz = mi * mj * i;
        sz = ni * nj * (i + ok) + oi;
        for (j = tidy; j < mj; j += stridey) {
            t = tz + mi * j;
            s = sz + ni * (j + oj);
            for (k = tidx; k < mi; k += stridex) {
                /*
                t = mi * mj * i;
                  + mi * j
                  + k
                s = ni * nj * (i + ok)
                  + ni * (j + oj)
                  + k + oi
                tgt[t] = src[s];
                */
                tgt[k + t] = src[k + s];
            }
        }
    }
}


int main(void)
{
    int i, j, k, s, t;
    const int dimx[3] = {100,100,100};
    const int dimy[3] = {10,10,10};
    const int n = dimx[0] * dimx[1] * dimx[2];
    const int m = dimy[0] * dimy[1] * dimy[2];
    int position[3] = {22,44,66};
    double x[n], y[m], y_ref[m];
    double *x_, *y_;

    dim3 blocks(32, 32, 32);
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
            for (k=0; k < dimy[2]; k++) {
                t = dimy[2] * dimy[1] * i
                  + dimy[2] * j
                  + k;
                s = dimx[2] * dimx[1] * (i + position[0])
                  + dimx[2] * (j + position[1])
                  + k + position[2];
                y_ref[t] = x[s];
            }
        }
    }

    // allocate + copy initial values
    cudaMalloc((void **) &x_, sizeof(double) * n);
    cudaMalloc((void **) &y_, sizeof(double) * m);
    cudaMemcpy(x_, x, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sizeof(double) * m, cudaMemcpyHostToDevice);

    // copy subarray on GPU
    copy3d_<<<blocks, threads>>>(x_, y_,
                                 dimx[0], dimx[1], dimx[2],
                                 dimy[0], dimy[1], dimy[2],
                                 position[0], position[1], position[2]);

    // copy result back to host and print with reference
    printf("\nC ordered\n---------\n");
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(y, y_, sizeof(double) * m, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);

    // copy reference values (Fortran ordered)
    for (i=0; i < dimy[2]; i++) {
        for (j=0; j < dimy[1]; j++) {
            for (k=0; k < dimy[0]; k++) {
                t = dimy[0] * dimy[1] * i
                  + dimy[0] * j
                  + k;
                s = dimx[0] * dimx[1] * (i + position[2])
                  + dimx[0] * (j + position[1])
                  + k + position[0];
                y_ref[t] = x[s];
            }
        }
    }

    // copy subarray on GPU (Fortran ordered)
    copy3d_fortran_<<<blocks, threads>>>(x_, y_,
                                 dimx[0], dimx[1], dimx[2],
                                 dimy[0], dimy[1], dimy[2],
                                 position[0], position[1], position[2]);

    // copy result back to host and print with reference
    printf("\nFortran ordered\n---------------\n");
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);
    cudaMemcpy(y, y_, sizeof(double) * m, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[m-2], y_ref[m-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[m-2], y[m-1]);

    return 0;
}
