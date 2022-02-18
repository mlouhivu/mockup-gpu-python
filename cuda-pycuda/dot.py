import atexit
from math import sin, cos

from numpy import array
from pycuda import driver
from pycuda import gpuarray

from _dot import dot

blocks = 32
threads = 256
n = 10000

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
z_ref = sum(x * y)

# allocate + copy initial values
x_ = gpuarray.to_gpu(x)
y_ = gpuarray.to_gpu(y)
z_ = gpuarray.empty(1, float)

# calculate dot product on GPU (and reduce partial sums)
dot(blocks, threads, n, x_.gpudata, y_.gpudata, z_.gpudata)

# copy result back to host and print with reference
z = z_.get()[0]
print(' reference: {0}'.format(z_ref))
print('    result: {0}'.format(z))
