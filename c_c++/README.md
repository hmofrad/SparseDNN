# SparseDNN
Sparse Deep Neural Network Training and Building written in C/C++.

## Build
    make clean && make

## Run
    export OMP_NUM_THREADS=12
    export OMP_PLACES=cores
    export OMP_PROC_BIND=close
    ./main -n 1024 -l 120 ../data/MNIST/ ../data/DNN/

## Contact
    Mohammad Hasanzadeh Mofrad
    m.hasanzadeh.mofrad@gmail.com
