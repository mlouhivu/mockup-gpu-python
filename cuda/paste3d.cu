#include <stdio.h>
#include <cuda_runtime.h>

__global__ void paste3d_(double *src, double *tgt,
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

    for (i = tidz; i < ni; i += stridez) {
        sz = nk * nj * i;
        tz = mk * mj * (i + oi) + ok;
        for (j = tidy; j < nj; j += stridey) {
            s = sz + nk * j;
            t = tz + mk * (j + oj);
            for (k = tidx; k < nk; k += stridex) {
                tgt[k + t] = src[k + s];
            }
        }
    }
}


__global__ void paste3d_fortran_(double *src, double *tgt,
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

    for (i = tidz; i < nk; i += stridez) {
        sz = ni * nj * i;
        tz = mi * mj * (i + ok) + oi;
        for (j = tidy; j < nj; j += stridey) {
            s = sz + ni * j;
            t = tz + mi * (j + oj);
            for (k = tidx; k < ni; k += stridex) {
                tgt[k + t] = src[k + s];
            }
        }
    }
}


int main(void)
{
    int i, j, k, s, t;
    const int dimx[3] = {10,10,10};
    const int dimy[3] = {100,100,100};
    const int n = dimx[0] * dimx[1] * dimx[2];
    const int m = dimy[0] * dimy[1] * dimy[2];
    int position[3] = {22,44,66};
    int start, end;
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
    for (i=0; i < dimx[0]; i++) {
        for (j=0; j < dimx[1]; j++) {
            for (k=0; k < dimx[2]; k++) {
                s = dimx[2] * dimx[1] * i
                  + dimx[2] * j
                  + k;
                t = dimy[2] * dimy[1] * (i + position[0])
                  + dimy[2] * (j + position[1])
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

    // copy into subarray on GPU
    paste3d_<<<blocks, threads>>>(x_, y_,
                                  dimx[0], dimx[1], dimx[2],
                                  dimy[0], dimy[1], dimy[2],
                                  position[0], position[1], position[2]);

    // copy result back to host and print with reference
    printf("\nC ordered\n---------\n");
    start = dimy[2] * dimy[1] * position[0]
          + dimy[2] * position[1]
          + position[2];
    end = dimy[2] * dimy[1] * (position[0] + dimx[0] - 1)
        + dimy[2] * (position[1] + dimx[1] - 1)
        + position[2] + dimx[2] - 1;
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[start], y[start+1], y[start+2], y[start+3],
            y[end-1], y[end]);
    cudaMemcpy(y, y_, sizeof(double) * m, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[start], y_ref[start+1], y_ref[start+2],
            y_ref[start+3], y_ref[end-1], y_ref[end]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[start], y[start+1], y[start+2], y[start+3],
            y[end-1], y[end]);

    // copy reference values (Fortran ordered)
    for (i=0; i < dimx[2]; i++) {
        for (j=0; j < dimx[1]; j++) {
            for (k=0; k < dimx[0]; k++) {
                s = dimx[0] * dimx[1] * i
                  + dimx[0] * j
                  + k;
                t = dimy[0] * dimy[1] * (i + position[2])
                  + dimy[0] * (j + position[1])
                  + k + position[0];
                y_ref[t] = x[s];
            }
        }
    }

    // copy into subarray on GPU (Fortran ordered)
    paste3d_fortran_<<<blocks, threads>>>(
            x_, y_, dimx[0], dimx[1], dimx[2], dimy[0], dimy[1], dimy[2],
            position[0], position[1], position[2]);

    // copy result back to host and print with reference
    printf("\nFortran ordered\n---------------\n");
    start = dimy[0] * dimy[1] * position[2]
          + dimy[0] * position[1]
          + position[0];
    end = dimy[0] * dimy[1] * (position[2] + dimx[2] - 1)
        + dimy[0] * (position[1] + dimx[1] - 1)
        + position[0] + dimx[0] - 1;
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[start], y[start+1], y[start+2], y[start+3],
            y[end-1], y[end]);
    cudaMemcpy(y, y_, sizeof(double) * m, cudaMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[start], y_ref[start+1], y_ref[start+2],
            y_ref[start+3], y_ref[end-1], y_ref[end]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[start], y[start+1], y[start+2], y[start+3],
            y[end-1], y[end]);

    return 0;
}
