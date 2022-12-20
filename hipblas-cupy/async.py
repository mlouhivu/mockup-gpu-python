from math import sin, cos
import timeit

from numpy import array
import cupy

from _async import daxpy_async
from _async import create_handle, destroy_handle
from _axpy import daxpy

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
t_cpu = timeit.repeat('axpy(a, x, y)',
                      number=repeat, repeat=1, globals=globals())

# create BLAS handle
handle = create_handle()

# calculate axpy on GPU and keep data on GPUs
t_async = timeit.repeat(
        'daxpy_async(handle, n, a_.data.ptr, x_.data.ptr, y_.data.ptr)',
        number=repeat, repeat=1, globals=globals())

# copy result back to host and print with reference
print('  initial: {0} {1} {2} {3} {4} {5}'.format(
    y_orig[0], y_orig[1], y_orig[2], y_orig[3], y_orig[-2], y_orig[-1]))
print('reference: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))
y_.get(out=y)
print('   result: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))

# calculate axpy on GPU without keeping data on GPUs
y[:] = y_orig[:]
y_ = cupy.asarray(y)
t_sync = timeit.repeat(
        'daxpy(n, a, x_.data.ptr, y_.data.ptr)',
        number=repeat, repeat=1, globals=globals())

# copy result back to host and print with reference
y_.get(out=y)
print('     sync: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))
# print timing info
print('----------')
print('time(CPU)  : {0:.6f}'.format(t_cpu[0]))
print('time(sync) : {0:.6f}'.format(t_sync[0]))
print('time(async): {0:.6f}'.format(t_async[0]))

# destroy BLAS handle
destroy_handle(handle)
del handle
