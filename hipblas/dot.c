#include <stdio.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <hipblas.h>

int main(void)
{
    int i;
    const int n = 100;
    double x[n], y[n];
    double *x_, *y_, *z_;
    double z_ref = 0.0;
    double z;
    hipblasHandle_t handle;

    // initialise data and calculate reference value on CPU
    for (i=0; i < n; i++) {
        x[i] = sin(i) * 2.3;
        y[i] = cos(i) * 1.1;
        z_ref += x[i] * y[i];
    }
    hipblasCreate(&handle);

    // use GPU pointers to keep data on GPU
    hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);

    // allocate + copy initial values
    hipMalloc((void **)(&x_), sizeof(double) * n);
    hipMalloc((void **)(&y_), sizeof(double) * n);
    hipMalloc((void **)(&z_), sizeof(double));
    hipMemcpy(x_, x, sizeof(double) * n, hipMemcpyHostToDevice);
    hipMemcpy(y_, y, sizeof(double) * n, hipMemcpyHostToDevice);

    // calculate dot on GPU
    hipblasDdot(handle, n, x_, 1, y_, 1, z_);

    // copy result back to host and print with reference
    hipMemcpy(&z, z_, sizeof(double), hipMemcpyDeviceToHost);
    printf("reference: %f\n   result: %f\n", z_ref, z);

    hipblasDestroy(handle);
    return 0;
}
