#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <hipblas.h>


void dot(hipblasHandle_t handle, int n, double *x_, double *y_, double *z_)
{
    hipblasDdot(handle, n, x_, 1, y_, 1, z_);
}

PyObject* dot_wrapper(PyObject *self, PyObject *args)
{
    int blocks;
    int threads;
    int n;
    double *x_;
    double *y_;
    double *z_;
    double *buffer_;
    hipblasHandle_t handle;

    if (!PyArg_ParseTuple(args, "iiinnn",
                          &blocks, &threads, &n, &x_, &y_, &z_))
        return NULL;

    hipblasCreate(&handle);
    dot(handle, n, x_, y_, z_);
    hipblasDestroy(handle);

    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"dot", dot_wrapper, METH_VARARGS, "dot"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "dot",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC
PyInit__dot(void)
{
    return PyModule_Create(&module);
}
