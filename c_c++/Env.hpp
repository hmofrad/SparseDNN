/*
 * Env.cpp: Multithreading environment 
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
#ifndef ENV_HPP
#define ENV_HPP

#include <omp.h>

class Env {
    public:
        Env();
        static int nthreads;
        static std::vector<uint64_t> start_col;
        static std::vector<uint64_t> end_col;
        static std::vector<uint64_t> start_nnz;
        static std::vector<uint64_t> end_nnz;
        static std::vector<uint64_t> length_nnz;
        static std::vector<uint64_t> offset_nnz;
        static std::vector<uint64_t> indices_nnz;

        static void init();
        static int env_get_num_threads();
        static void env_unset(int tid);
        static uint64_t env_set();
};
int Env::nthreads = 0;
std::vector<uint64_t> Env::start_col;
std::vector<uint64_t> Env::end_col;
std::vector<uint64_t> Env::start_nnz;
std::vector<uint64_t> Env::end_nnz;
std::vector<uint64_t> Env::length_nnz;
std::vector<uint64_t> Env::offset_nnz;
std::vector<uint64_t> Env::indices_nnz;

void Env::init() {
    nthreads = env_get_num_threads();
    start_col.resize(nthreads);
    end_col.resize(nthreads);
    start_nnz.resize(nthreads);
    end_nnz.resize(nthreads);
    length_nnz.resize(nthreads);
    offset_nnz.resize(nthreads);
    indices_nnz.resize(nthreads);
}

int Env::env_get_num_threads() {
    int nthreads_ = 0;
    #pragma omp parallel
    {
        nthreads_ = omp_get_num_threads();
    }
    return(nthreads_);
}    

void Env::env_unset(int tid) {    
    //for(uint32_t i = 0; i < Env::nthreads; i++) {   
        start_col[tid] = 0;
        end_col[tid] = 0;
        start_nnz[tid] = 0;
        end_nnz[tid] = 0;
        offset_nnz[tid] = 0;
        indices_nnz[tid] = 0;
        length_nnz[tid] = 0;
    //}
}

uint64_t Env::env_set() {  
    for(uint32_t i = 0; i < Env::nthreads; i++) {   
        start_nnz[i] = 0;
        end_nnz[i] = 0;
        offset_nnz[i] = 0;
        indices_nnz[i] = 0;
    }
    offset_nnz[0] = 0;
    start_nnz[0] = 0;
    end_nnz[0] = length_nnz[0];
    uint64_t nnzmax = length_nnz[0];
    for(uint32_t i = 1; i < Env::nthreads; i++) {
        start_nnz[i] = end_nnz[i-1];
        end_nnz[i] = start_nnz[i] + length_nnz[i];
        offset_nnz[i] = start_nnz[i];
        nnzmax += length_nnz[i];
    }
    return(nnzmax);
}

#endif 