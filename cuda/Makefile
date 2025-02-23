NVCCFLAGS = -g -G

all: build

build: matrix_add.out vector_add.out vector_add_simple.out

matrix_add.out: matrix_add.cu
	nvcc $(NVCCFLAGS) -o matrix_add.out matrix_add.cu

vector_add.out: vector_add.cu
	nvcc $(NVCCFLAGS) -o vector_add.out vector_add.cu

vector_add_simple.out: vector_add_simple.cu
	nvcc $(NVCCFLAGS) -o vector_add_simple.out vector_add_simple.cu

clean:
	rm -f vector_add.out
	rm -f matrix_add.out
	rm -f vector_add_simple.out

run: build
	./vector_add.out
	./matrix_add.out
	./vector_add_simple.out