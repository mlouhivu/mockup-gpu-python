#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void sum_(int n, double *x, double *y)
{
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    extern __shared__ double buffer[];
    int i;

    double sum = 0.0f;
    for (i = gid; i < n; i += stride) {
        sum += x[i];
    }
    buffer[tid] = sum;
    __syncthreads();

    i = blockDim.x / 2;
    while (i > 0) {
        if (tid < i)
            buffer[tid] += buffer[tid + i];
        i = i / 2;
        __syncthreads();
    }

    if (tid == 0)
        y[blockIdx.x] = buffer[0];
}

__global__ void dot_(int n, double *x, double *y, double *z)
{
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    extern __shared__ double buffer[];
    int i;

    double sum = 0.0f;
    for (i = gid; i < n; i += stride) {
        sum += x[i] * y[i];
    }
    buffer[tid] = sum;
    __syncthreads();

    i = blockDim.x / 2;
    while (i > 0) {
        if (tid < i)
            buffer[tid] += buffer[tid + i];
        i = i / 2;
        __syncthreads();
    }

    if (tid == 0)
        z[blockIdx.x] = buffer[0];
}

void sum(int blocks, int threads, int shmem,
         int n, double *x_, double *y_)
{
    sum_<<<blocks, threads, shmem>>>(n, x_, y_);
}

void dot(int blocks, int threads, int shmem,
         int n, double *x_, double *y_, double *z_)
{
    dot_<<<blocks, threads, shmem>>>(n, x_, y_, z_);
}

extern "C"
PyObject* dot_wrapper(PyObject *self, PyObject *args)
{
    int blocks;
    int threads;
    int n;
    PyObject *xc;
    PyObject *yc;
    PyObject *zc;
    double *x_;
    double *y_;
    double *z_;
    double *buffer_;

    if (!PyArg_ParseTuple(args, "iiiOOO",
                          &blocks, &threads, &n, &xc, &yc, &zc))
        return NULL;
    x_ = (double *) PyCapsule_GetPointer(xc, NULL);
    y_ = (double *) PyCapsule_GetPointer(yc, NULL);
    z_ = (double *) PyCapsule_GetPointer(zc, NULL);

    cudaMalloc((void **) &buffer_, sizeof(double) * blocks);

    dot(blocks, threads, threads * sizeof(double), n, x_, y_, buffer_);
    sum(1, blocks, blocks * sizeof(blocks), blocks, buffer_, z_);

    Py_RETURN_NONE;
}

extern "C"
static PyMethodDef methods[] = {
    {"dot", dot_wrapper, METH_VARARGS, "dot"},
    {NULL, NULL, 0, NULL}
    };
extern "C"
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "dot",
    NULL,
    -1,
    methods
};

extern "C"
PyMODINIT_FUNC
PyInit__dot(void)
{
    return PyModule_Create(&module);
}
