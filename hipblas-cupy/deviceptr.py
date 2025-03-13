from math import sin, cos

from numpy import array
import cupy

from _deviceptr import daxpy
from _deviceptr import create_handle, destroy_handle

n = 10000
a = 3.4

def axpy(a, x, y):
    y += a * x

# initialise data
x = array([sin(i) * 2.3 for i in range(n)], float)
y = array([cos(i) * 1.1 for i in range(n)], float)
y_orig = y.copy()

# allocate + copy initial values
x_ = cupy.asarray(x)
y_ = cupy.asarray(y)
a_ = cupy.asarray(array([a], float))

# calculate reference values on CPU
axpy(a, x, y)
print('  initial: {0} {1} {2} {3} {4} {5}'.format(
    y_orig[0], y_orig[1], y_orig[2], y_orig[3], y_orig[-2], y_orig[-1]))
print('reference: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))

# create BLAS handle
create_handle()

# calculate axpy on GPU
daxpy(n, a_.data.ptr, x_.data.ptr, y_.data.ptr)
y_.get(out=y)
print('   global: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))

# destroy BLAS handle
destroy_handle()
