#!/bin/make
# Makefile
# (c) Mohammad Hasanzadeh Mofrad, 2019
# (e) m.hasanzadeh.mofrad@gmail.com

OBJ=baseline
CXX = g++
CXX_FLAGS = -std=c++14
CXX_OPT = -DNDEBUG -O3 -flto -fwhole-program -march=native -ftree-vectorize -ffast-math -funroll-loops

install:
	$(CXX) $(CXX_FLAGS) $(CXX_OPT) -o $(OBJ) $(OBJ).cpp 
clean:
	rm -rf $(OBJ)
