import atexit
from math import sin, cos

from numpy import array
from pycuda import driver
from pycuda import gpuarray

from _axpy import daxpy

n = 10000
a = 3.4

def detach(context):
    context.pop()
    context.detach()

# initialise CUDA
driver.init()
device = driver.Device(0)
context = device.make_context(flags=driver.ctx_flags.SCHED_YIELD)
context.set_cache_config(driver.func_cache.PREFER_L1)
context.push()
atexit.register(detach, context)

# initialise data and calculate reference values on CPU
x = array([sin(i) * 2.3 for i in range(n)], float)
y = array([cos(i) * 1.1 for i in range(n)], float)
y_ref = a * x + y

# allocate + copy initial values
x_ = gpuarray.to_gpu(x)
y_ = gpuarray.to_gpu(y)

# calculate axpy on GPU
daxpy(n, a, x_.gpudata, y_.gpudata)

# copy result back to host and print with reference
print('  initial: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))
y_.get(y)
print('reference: {0} {1} {2} {3} {4} {5}'.format(
    y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[-2], y_ref[-1]))
print('   result: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))
