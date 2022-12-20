#include <stdio.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <hipblas.h>

int main(void)
{
    int i;
    const int n = 100;
    double a = 3.4;
    double x[n], y[n], y_ref[n];
    double *x_, *y_;
    hipblasHandle_t handle;

    // initialise data and calculate reference values on CPU
    for (i=0; i < n; i++) {
        x[i] = sin(i) * 2.3;
        y[i] = cos(i) * 1.1;
        y_ref[i] = a * x[i] + y[i];
    }
    printf("  initial: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[n-2], y[n-1]);
    hipblasCreate(&handle);

    // allocate + copy initial values
    hipMalloc((void **) &x_, sizeof(double) * n);
    hipMalloc((void **) &y_, sizeof(double) * n);
    hipMemcpy(x_, x, sizeof(double) * n, hipMemcpyHostToDevice);
    hipMemcpy(y_, y, sizeof(double) * n, hipMemcpyHostToDevice);

    // calculate axpy on GPU
    hipblasDaxpy(handle, n, &a, x_, 1, y_, 1);

    // copy result back to host and print with reference
    hipMemcpy(&y, y_, sizeof(double) * n, hipMemcpyDeviceToHost);
    printf("reference: %f %f %f %f ... %f %f\n",
            y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[n-2], y_ref[n-1]);
    printf("   result: %f %f %f %f ... %f %f\n",
            y[0], y[1], y[2], y[3], y[n-2], y[n-1]);

    hipblasDestroy(handle);
    return 0;
}
