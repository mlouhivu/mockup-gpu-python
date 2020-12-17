#include <stdio.h>
#include <cuda_runtime.h>

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
void print_hello(int n, int *tag)
{
    int start, end;
    int span[n];

    range_finder(n, tag, (int *) span);
    int i = 0;
    while (i < n && span[i] != -1) {
        start = span[i];
        end = span[i+1];
        i += 2;
        if (start == end)
            printf("Hello from %d\n", start);
        else
            printf("Hello from {%d..%d}\n", start, end);
    }
}


int main(void)
{
    int i;
    const int maxsize = 65536;
    const int n = 10000;
    int tag[maxsize];
    int *tag_;

    dim3 blocks(32);
    dim3 threads(256);

    cudaMalloc((void **) &tag_, sizeof(int) * maxsize);

    // initialise tags
    for (i=0; i < maxsize; i++) {
        tag[i] = 0;
    }
    cudaMemcpy(tag_, tag, sizeof(int) * maxsize, cudaMemcpyHostToDevice);

    // flip tag for hello
    hello_single_<<<blocks, threads>>>(n, tag_);

    // print out the hellos
    cudaMemcpy(tag, tag_, sizeof(int) * maxsize, cudaMemcpyDeviceToHost);
    printf("\nSINGLE\n");
    print_hello(n, (int *) tag);

    // initialise tags
    for (i=0; i < maxsize; i++) {
        tag[i] = 0;
    }
    cudaMemcpy(tag_, tag, sizeof(int) * maxsize, cudaMemcpyHostToDevice);

    // flip tag for hello
    hello_<<<blocks, threads>>>(n, tag_);

    // print out the hellos
    cudaMemcpy(tag, tag_, sizeof(int) * maxsize, cudaMemcpyDeviceToHost);
    printf("\n1D\n");
    print_hello(n, (int *) tag);

    return 0;
}
