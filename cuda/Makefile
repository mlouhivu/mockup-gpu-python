CFLAGS := -gencode arch=compute_70,code=sm_70
CC := nvcc

exe := $(patsubst %.cu, %, $(wildcard *.cu))

all: $(exe)

%: %.cu
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm $(exe)
