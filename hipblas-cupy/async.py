from math import sin, cos

from numpy import array
import cupy

from _async import daxpy_async
from _async import create_handle, destroy_handle

n = 10000
a = 3.4
repeat = 5

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
for i in range(repeat):
    axpy(a, x, y)

# create BLAS handle
handle = create_handle()

# calculate axpy on GPU and keep data on GPUs
for i in range(repeat):
    daxpy_async(handle, n, a_.data.ptr, x_.data.ptr, y_.data.ptr)

# copy result back to host and print with reference
print('  initial: {0} {1} {2} {3} {4} {5}'.format(
    y_orig[0], y_orig[1], y_orig[2], y_orig[3], y_orig[-2], y_orig[-1]))
print('reference: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))
y_.get(out=y)
print('   result: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))

# destroy BLAS handle
destroy_handle(handle)
del handle
