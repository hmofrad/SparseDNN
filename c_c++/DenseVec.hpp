/*
 * DenseVec.hpp: Dense vector
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@pitt.edu
 */
 
#ifndef DENSEVEC_HPP
#define DENSEVEC_HPP

#include "Allocator.hpp"

template<typename Data_Type>
struct DenseVec {
    public: 
        DenseVec() {nitems = 0; nbytes = 0, A = nullptr;};
        DenseVec(uint32_t nitems_);
        ~DenseVec();
        void clear();
        void walk();
        uint64_t nitems;
        uint64_t nbytes;
        Data_Type *A;
        struct Data_Block<Data_Type> *A_blk;
};

template<typename Data_Type>
DenseVec<Data_Type>::DenseVec(uint32_t nitems_) {
    nitems = nitems_;
    A_blk = new Data_Block<Data_Type>(&A, nitems, nitems * sizeof(Data_Type));
    nbytes = A_blk->nbytes;
}

template<typename Data_Type>
DenseVec<Data_Type>::~DenseVec(){
    delete A_blk;
    A = nullptr;
}

template<typename Data_Type>
void DenseVec<Data_Type>::clear(){
    A_blk->clear();
}

template<typename Data_Type>
void DenseVec<Data_Type>::walk() {
    for(uint64_t i = 0; i < nitems; i++)
        std::cout << i << " " << A[i] << std::endl;
}



#endif