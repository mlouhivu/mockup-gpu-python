from math import sin, cos

from numpy import array
import cupy

from _async import daxpy_async

n = 10000
a = 3.4

# initialise data and calculate reference values on CPU
x = array([sin(i) * 2.3 for i in range(n)], float)
y = array([cos(i) * 1.1 for i in range(n)], float)
y_ref = a * x + y

# allocate + copy initial values
x_ = cupy.asarray(x)
y_ = cupy.asarray(y)
a_ = cupy.asarray(array([a], float))

# calculate axpy on GPU and keep data on GPUs
daxpy_async(n, a_.data.ptr, x_.data.ptr, y_.data.ptr)

# copy result back to host and print with reference
print('  initial: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))
y_.get(out=y)
print('reference: {0} {1} {2} {3} {4} {5}'.format(
    y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[-2], y_ref[-1]))
print('   result: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))
