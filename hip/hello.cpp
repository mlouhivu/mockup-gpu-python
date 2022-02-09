#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void hello_(int n, int *tag)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        tag[tid] = 1;
    }
}

__global__ void hello_single_(int n, int *tag)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n)
        tag[tid] = 1;
}

__global__ void hello2D_(int n, int2 dim, int *tag, int3 *coord)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int stridex = gridDim.x * blockDim.x;
    int stridey = gridDim.y * blockDim.y;
    int gid;
    int i, j;

    for (i = tidy; i < dim.x; i += stridey) {
        for (j = tidx; j < dim.y; j += stridex) {
            gid = dim.y * i + j;
            if (gid < n) {
                tag[gid] = 1;
                coord[gid].x = i;
                coord[gid].y = j;
            }
        }
    }
}

__global__ void hello2D_single_(int n, int *tag, int3 *coord)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int gid = gridDim.x * blockDim.x * tidy + tidx;

    if (gid < n) {
        tag[gid] = 1;
        coord[gid].x = tidy;
        coord[gid].y = tidx;
    }
}

/*
   Find continuous ranges of non-zero values in buffer. Stores start and
   end indeces in span.
*/
void range_finder(int n, int *buffer, int *span)
{
    int on = 0;
    int j = 0;
    for (int i=0; i < n; i++) {
        span[i] = -1;
        if (buffer[i]) {
            if (!on) {
                span[j] = i;
                j++;
            }
            on = 1;
        } else if (on) {
            span[j] = i - 1;
            j++;
            on = 0;
        }
    }
    if (on) {
        span[j] = n - 1;
    }
}

/*
   Print hellos from non-zero tags, coalescing continuous ranges.
*/
void print_hello(int n, int *tag, int3 *coord, int dim)
{
    int start, end;
    int span[n];
    char str[1024];

    range_finder(n, tag, (int *) span);
    int i = 0;
    while (i < n && span[i] != -1) {
        start = span[i];
        end = span[i+1];
        i += 2;
        if (dim == 2) {
            sprintf(str, "  (%d %d)..(%d %d)",
                    coord[start].x, coord[start].y,
                    coord[end].x, coord[end].y);
        } else {
            sprintf(str, "");
        }

        if (start == end)
            printf("Hello from %d%s\n", start, str);
        else
            printf("Hello from {%d..%d}%s\n", start, end, str);
    }
}


int main(void)
{
    int i;
    const int maxsize = 65536;
    const int n = 10000;
    int2 size2 = {1,10000};
    int tag[maxsize];
    int *tag_;
    int3 coord[maxsize];
    int3 *coord_;

    dim3 blocks(32);
    dim3 threads(256);
    dim3 blocks2(32,32);
    dim3 threads2(16,16);

    hipMalloc((void **) &tag_, sizeof(int) * maxsize);
    hipMalloc((void **) &coord_, sizeof(int3) * maxsize);

    // initialise tags
    for (i=0; i < maxsize; i++) {
        tag[i] = 0;
    }
    hipMemcpy(tag_, tag, sizeof(int) * maxsize, hipMemcpyHostToDevice);

    // simple 1D kernel (single thread, single operation)
    hipLaunchKernelGGL(hello_single_, dim3(blocks), dim3(threads), 0, 0, n, tag_);

    // print out the hellos
    hipMemcpy(tag, tag_, sizeof(int) * maxsize, hipMemcpyDeviceToHost);
    printf("\nSINGLE\n");
    print_hello(n, (int *) tag, coord, 1);

    // initialise tags
    for (i=0; i < maxsize; i++) {
        tag[i] = 0;
    }
    hipMemcpy(tag_, tag, sizeof(int) * maxsize, hipMemcpyHostToDevice);

    // flexible 1D kernel (single thread, multiple operations)
    hipLaunchKernelGGL(hello_, dim3(blocks), dim3(threads), 0, 0, n, tag_);

    // print out the hellos
    hipMemcpy(tag, tag_, sizeof(int) * maxsize, hipMemcpyDeviceToHost);
    printf("\n1D\n");
    print_hello(n, (int *) tag, coord, 1);

    // initialise tags + coords
    for (i=0; i < maxsize; i++) {
        tag[i] = 0;
        coord[i].x = -1;
        coord[i].y = -1;
        coord[i].z = -1;
    }
    hipMemcpy(tag_, tag, sizeof(int) * maxsize, hipMemcpyHostToDevice);
    hipMemcpy(coord_, coord, sizeof(int3) * maxsize, hipMemcpyHostToDevice);

    // simple 2D kernel
    hipLaunchKernelGGL(hello2D_single_, dim3(blocks2), dim3(threads2), 0, 0, n, tag_, coord_);

    // print out the hellos
    hipMemcpy(tag, tag_, sizeof(int) * maxsize, hipMemcpyDeviceToHost);
    hipMemcpy(coord, coord_, sizeof(int3) * maxsize, hipMemcpyDeviceToHost);
    printf("\n2D SINGLE\n");
    print_hello(n, (int *) tag, coord, 2);

    // initialise tags + coords
    for (i=0; i < maxsize; i++) {
        tag[i] = 0;
        coord[i].x = -1;
        coord[i].y = -1;
        coord[i].z = -1;
    }
    hipMemcpy(tag_, tag, sizeof(int) * maxsize, hipMemcpyHostToDevice);
    hipMemcpy(coord_, coord, sizeof(int3) * maxsize, hipMemcpyHostToDevice);

    // flexible 2D kernel
    hipLaunchKernelGGL(hello2D_, dim3(blocks2), dim3(threads2), 0, 0, n, size2, tag_, coord_);

    // print out the hellos
    hipMemcpy(tag, tag_, sizeof(int) * maxsize, hipMemcpyDeviceToHost);
    hipMemcpy(coord, coord_, sizeof(int3) * maxsize, hipMemcpyDeviceToHost);
    printf("\n2D\n");
    print_hello(n, (int *) tag, coord, 2);

    return 0;
}
