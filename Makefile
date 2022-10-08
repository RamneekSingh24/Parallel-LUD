CC := /opt/homebrew/Cellar/gcc/12.2.0/bin/g++-12

all: pthread openmp

pthread:
	$(CC) -std=c++11 -march=native pthread_LU.cpp -g -lpthread -O3 -ftree-vectorizer-verbose=1 -DCACHE_BLOCK_SIZE=$$(cat /proc/cpuinfo | grep cache_alignment | head -1 |sed 's/^.*: //') -o lupt
	
openmp:
	$(CC) -std=c++11 -march=native pthread_LU.cpp -g -fopenmp -O3 -ftree-vectorizer-verbose=1 -DCACHE_BLOCK_SIZE=$$(cat /proc/cpuinfo | grep cache_alignment | head -1 |sed 's/^.*: //') -o luomp

simd: simdtest.cpp
	$(CC) -std=c++11 -march=native simdtest.cpp -g -fopenmp -o simd

clean:
	rm luomp lupt	



