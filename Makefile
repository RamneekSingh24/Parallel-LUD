all: pthread openmp openmp_b

pthread:
	g++ -std=c++11 -march=native pthread_LU.cpp -g -lpthread -O3 -ffast-math -o lupt
	
openmp:
	g++ -std=c++11 -march=native openmp_LU.cpp -g -fopenmp -O3 -ffast-math -o luomp

openmp_b:
	g++ -std=c++11 -march=native openmp_LU_block.cpp -g -fopenmp -O3 -ffast-math -o lubomp

clean:
	rm luomp lupt lubomp



