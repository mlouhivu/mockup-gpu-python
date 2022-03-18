import numpy as np
import cupy as cp

# Note that evolve_buffers require cupy with Python buffer protocol support
# (see e.g. https://github.com/cupy/cupy/issues/1532 how to add manually)
# evolve_pointer works with vanilla cupy
from _evolve import evolve_buffer, evolve_buffer_gpu, evolve_pointer

# Use managed memory
cp.cuda.set_allocator(cp.cuda.malloc_managed)

# sin field
x = np.linspace(-np.pi, np.pi, 1000)
y = np.linspace(-np.pi, np.pi, 1000)
xy = np.meshgrid(x, y)
field = np.sin(xy)
field0 = field.copy()

# GPU arrays
field_gpu = cp.asarray(field)
field0_gpu = field_gpu.copy()

field2_gpu = cp.asarray(field)
field2_0_gpu = field_gpu.copy()

# By default, gpu-to-gpu memcpy is asynchronous, without explicit
# synchronization computation may start before memcpy is complete
cp.cuda.get_current_stream().synchronize()

a = 0.5
dx2 = 0.01**2
dy2 = 0.01**2
dt = dx2*dy2 / ( 2*a*(dx2+dy2) )

def evolve_py(u, u_previous, a, dt, dx2, dy2):
    u[1:-1, 1:-1] = u_previous[1:-1, 1:-1] + a * dt * ( \
            (u_previous[2:, 1:-1] - 2*u_previous[1:-1, 1:-1] + \
             u_previous[:-2, 1:-1]) / dx2 + \
            (u_previous[1:-1, 2:] - 2*u_previous[1:-1, 1:-1] + \
                 u_previous[1:-1, :-2]) / dy2 )
    u_previous[:] = u[:]

def evolve_np(u, u_previous, a, dt, dx2, dy2):
    nx, ny = u.shape[0], u.shape[1]
    evolve_pointer(u.__array_interface__['data'][0], u_previous.__array_interface__['data'][0], nx, ny, a, dt, dx2, dy2)

def evolve_cp_cpu(u, u_previous, a, dt, dx2, dy2):
    nx, ny = field.shape[0], field.shape[1]
    evolve_pointer(u.data.ptr, u_previous.data.ptr, nx, ny, a, dt, dx2, dy2)

# Iterate with NumPy
for it in range(200):
    evolve_buffer(field, field0, a, dt, dx2, dy2)

# Iterate in GPU
for it in range(200):
    evolve_buffer_gpu(field_gpu, field0_gpu, a, dt, dx2, dy2)

max_dif = np.max(np.abs(field - field_gpu.get()))

print("{:^19} {:^19} {:^7}".format("NumPy", "GPU", "max_dif"))
print("{:19.17f} {:19.17f} {:19.17f}".format((field**2).mean(), (field_gpu**2).mean(), max_dif))

# GPU array with managed memory, computation in CPU
for it in range(200):
    evolve_buffer(field2_gpu, field2_0_gpu, a, dt, dx2, dy2)

max_dif = np.max(np.abs(field - field2_gpu.get()))

print("{:^19} {:^19} {:^7}".format("NumPy", "CPU + GPU array", "max_dif"))
print("{:19.17f} {:19.17f} {:19.17f}".format((field**2).mean(), (field2_gpu**2).mean(), max_dif))
