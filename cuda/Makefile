CFLAGS := -gencode arch=compute_70,code=sm_70
CC := nvcc

all: axpy axpy-mono dot sum copy copy2d copy3d paste paste2d paste3d

%: %.cu
	$(CC) $(CFLAGS) $< -o $@
