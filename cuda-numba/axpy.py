from math import sin, cos

import numpy
import numba
import numba.cuda

from _axpy import daxpy

n = 10000
a = 3.4

# initialise data and calculate reference values on CPU
x = numpy.array([sin(i) * 2.3 for i in range(n)], float)
y = numpy.array([cos(i) * 1.1 for i in range(n)], float)
y_ref = a * x + y

# allocate + copy initial values
x_ = numba.cuda.to_device(x)
y_ = numba.cuda.to_device(y)

# calculate axpy on GPU
daxpy(n, a, x_.device_ctypes_pointer.value, y_.device_ctypes_pointer.value)

# copy result back to host and print with reference
print('  initial: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))
y_.copy_to_host(y)
print('reference: {0} {1} {2} {3} {4} {5}'.format(
    y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[-2], y_ref[-1]))
print('   result: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))
