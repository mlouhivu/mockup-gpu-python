import numpy
from pycuda import driver
from pycuda import gpuarray
from pycuda.compiler import SourceModule

from basic import numpify

def axpy(dtype=float):
    src = SourceModule("""
    __global__ void daxpy(double a, double *x, double *y)
    {
        const int i = threadIdx.x;
        y[i] += a * x[i];
    }

    __global__ void saxpy(float a, float *x, float *y)
    {
        const int i = threadIdx.x;
        y[i] = a * x[i] + y[i];
    }
    """)

    if dtype in [float, numpy.float64]:
        print('Double precision AXPY')
        axpy = src.get_function('daxpy')
        a = numpy.float64(3.4)
    else:
        print('Single precision AXPY')
        axpy = src.get_function('saxpy')
        a = numpy.float32(3.4)
    x = numpify(None, dtype=dtype)
    y = numpify(None, dtype=dtype)

    print('Before axpy')
    print('a = {0}'.format(a))
    print('x = {0}'.format(x))
    print('y = {0}'.format(y))

    x_ = gpuarray.to_gpu(x)
    y_ = gpuarray.to_gpu(y)
    axpy(a, x_, y_, block=(len(x),1,1))
    x = x_.get()
    y = y_.get()

    # equivalent one-liner for one-off offloading:
    #   axpy(a, driver.In(x), driver.InOut(y), block=(len(x),1,1))

    print('After axpy')
    print('x = {0}'.format(x))
    print('y = {0}'.format(y))
