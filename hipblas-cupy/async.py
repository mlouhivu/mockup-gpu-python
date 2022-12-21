from math import sin, cos
import timeit

from numpy import array
import cupy

from _async import daxpy_global, daxpy_capsule
from _async import create_global, destroy_global
from _async import create_capsule, destroy_capsule
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
print('  initial: {0} {1} {2} {3} {4} {5}'.format(
    y_orig[0], y_orig[1], y_orig[2], y_orig[3], y_orig[-2], y_orig[-1]))
print('reference: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))

# create BLAS handle (in a capsule)
handle = create_capsule()

# calculate axpy on GPU asynchronously using capsules
t_capsule = timeit.repeat(
        'daxpy_capsule(handle, n, a_.data.ptr, x_.data.ptr, y_.data.ptr)',
        number=repeat, repeat=1, globals=globals())
y_.get(out=y)
print('  capsule: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))

# destroy BLAS handle
destroy_capsule(handle)
del handle

# restore original values
y[:] = y_orig[:]
y_ = cupy.asarray(y)

# create BLAS handle (global variable)
create_global()

# calculate axpy on GPU asynchronously using a global handle
t_global = timeit.repeat(
        'daxpy_global(n, a_.data.ptr, x_.data.ptr, y_.data.ptr)',
        number=repeat, repeat=1, globals=globals())
y_.get(out=y)
print('   global: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))

# destroy BLAS handle
destroy_global()

# restore original values
y[:] = y_orig[:]
y_ = cupy.asarray(y)

# calculate axpy on GPU synchronously
t_sync = timeit.repeat(
        'daxpy(n, a, x_.data.ptr, y_.data.ptr)',
        number=repeat, repeat=1, globals=globals())
y_.get(out=y)
print('     sync: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))

# print timing info
print('----------')
print('time(CPU):     {0:.6f}'.format(t_cpu[0]))
print('time(sync):    {0:.6f}'.format(t_sync[0]))
print('time(global):  {0:.6f}'.format(t_global[0]))
print('time(capsule): {0:.6f}'.format(t_capsule[0]))
