CFLAGS := -shared -I$(PYTHONHOME)/include/python3.8 -fPIC
CC := hipcc

exe := $(patsubst %.cpp, _%.so, $(wildcard *.cpp))

all: $(exe)

_%.so: %.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm $(exe)
