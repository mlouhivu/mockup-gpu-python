import numpy
import numba
from numba import cuda

from basic import numpify

@numba.cuda.jit
def _axpy(a, x, y):
    i = numba.cuda.threadIdx.x
    y[i] += a * x[i]


def axpy(dtype=float):
    if dtype in [float, numpy.float64]:
        print('Double precision AXPY')
        a = numpy.float64(3.4)
    else:
        print('Single precision AXPY')
        a = numpy.float32(3.4)
    x = numpify(None, dtype=dtype)
    y = numpify(None, dtype=dtype)

    print('Before axpy')
    print('a = {0}'.format(a))
    print('x = {0}'.format(x))
    print('y = {0}'.format(y))
    print('')

    x_ = numba.cuda.to_device(x)
    y_ = numba.cuda.to_device(y)

    print('Reference')
    print('y = {0}'.format(a * x + y))
    print('')

    _axpy[(1,1), (len(x),1)](a, x_, y_)
    x = x_.copy_to_host()
    y = y_.copy_to_host()

    print('After axpy')
    print('x = {0}'.format(x))
    print('y = {0}'.format(y))

if __name__ == '__main__':
    axpy()
