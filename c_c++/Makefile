#!/bin/make
# Makefile
# (c) Mohammad Hasanzadeh Mofrad, 2019
# (e) m.hasanzadeh.mofrad@gmail.com

OBJ=baseline
CXX = g++
CXX_FLAGS = -std=c++14
CXX_OPT = -DNDEBUG -O3 -flto -fwhole-program -march=native -ftree-vectorize -ffast-math -funroll-loops
THREADED = -fopenmp -D_GLIBCXX_PARALLEL -pthread

install:
	$(CXX) $(CXX_FLAGS) $(CXX_OPT) $(THREADED) -o $(OBJ) $(OBJ).cpp 
clean:
	rm -rf $(OBJ)
