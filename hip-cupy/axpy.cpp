#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void daxpy_(int n, double a, double *x, double *y)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        y[tid] += a * x[tid];
    }
}

__global__ void saxpy_(int n, float a, float *x, float *y)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        y[tid] += a * x[tid];
    }
}

void daxpy(int n, double a, double *x_, double *y_)
{
    hipLaunchKernelGGL(daxpy_, 32, 256, 0, 0, n, a, x_, y_);
}

void saxpy(int n, float a, float *x_, float *y_)
{
    hipLaunchKernelGGL(saxpy_, 32, 256, 0, 0, n, a, x_, y_);
}

extern "C"
PyObject* daxpy_wrapper(PyObject *self, PyObject *args)
{
    int n;
    double a;
    double *x_;
    double *y_;

    if (!PyArg_ParseTuple(args, "idnn", &n, &a, &x_, &y_))
        return NULL;

    daxpy(n, a, x_, y_);

    Py_RETURN_NONE;
}

extern "C"
PyObject* saxpy_wrapper(PyObject *self, PyObject *args)
{
    int n;
    float a;
    float *x_;
    float *y_;

    if (!PyArg_ParseTuple(args, "ifnn", &n, &a, &x_, &y_))
        return NULL;

    saxpy(n, a, x_, y_);

    Py_RETURN_NONE;
}

extern "C"
static PyMethodDef methods[] = {
    {"daxpy",  daxpy_wrapper, METH_VARARGS, "Daxpy"},
    {"saxpy",  saxpy_wrapper, METH_VARARGS, "Saxpy"},
    {NULL, NULL, 0, NULL}
};

extern "C"
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "axpy",
    NULL,
    -1,
    methods
};

extern "C"
PyMODINIT_FUNC
PyInit__axpy(void)
{
    return PyModule_Create(&module);
}
