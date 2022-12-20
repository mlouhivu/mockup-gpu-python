#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <hipblas.h>


void daxpy(hipblasHandle_t handle, int n, double *a_, double *x_, double *y_)
{
    hipblasDaxpy(handle, n, a_, x_, 1, y_, 1);
}

void saxpy(hipblasHandle_t handle, int n, float *a_, float *x_, float *y_)
{
    hipblasSaxpy(handle, n, a_, x_, 1, y_, 1);
}

PyObject* daxpy_wrapper(PyObject *self, PyObject *args)
{
    PyObject *c;
    int n;
    double *a_;
    double *x_;
    double *y_;
    hipblasHandle_t *handle;

    if (!PyArg_ParseTuple(args, "Oinnn", &c, &n, &a_, &x_, &y_))
        return NULL;
    handle = PyCapsule_GetPointer(c, NULL);

    daxpy(*handle, n, a_, x_, y_);

    Py_RETURN_NONE;
}

PyObject* saxpy_wrapper(PyObject *self, PyObject *args)
{
    PyObject *c;
    int n;
    float *a_;
    float *x_;
    float *y_;
    hipblasHandle_t *handle;

    if (!PyArg_ParseTuple(args, "Oinnn", &c, &n, &a_, &x_, &y_))
        return NULL;
    handle = PyCapsule_GetPointer(c, NULL);

    saxpy(*handle, n, a_, x_, y_);

    Py_RETURN_NONE;
}

PyObject* create_handle(PyObject *self, PyObject *args)
{
    hipblasHandle_t *handle;

    handle = malloc(sizeof(hipblasHandle_t));
    hipblasCreate(handle);

    // use GPU pointers to keep data on GPU
    hipblasSetPointerMode(*handle, HIPBLAS_POINTER_MODE_DEVICE);

    return PyCapsule_New(handle, NULL, NULL);
}

PyObject* destroy_handle(PyObject *self, PyObject *args)
{
    PyObject *c;
    hipblasHandle_t *handle;

    if (!PyArg_ParseTuple(args, "O", &c))
        return NULL;
    handle = PyCapsule_GetPointer(c, NULL);

    hipDeviceSynchronize();
    hipblasDestroy(*handle);
    free(handle);

    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"daxpy_async",    daxpy_wrapper, METH_VARARGS, "Daxpy"},
    {"saxpy_async",    saxpy_wrapper, METH_VARARGS, "Saxpy"},
    {"create_handle",  create_handle, METH_VARARGS, "Create hipBLAS handle"},
    {"destroy_handle", destroy_handle, METH_VARARGS, "Destroy hipBLAS handle"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "async",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC
PyInit__async(void)
{
    return PyModule_Create(&module);
}
