CFLAGS := 
CC := hipcc

exe := $(patsubst %.cpp, %, $(wildcard *.cpp))

all: $(exe)

%: %.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm $(exe)
