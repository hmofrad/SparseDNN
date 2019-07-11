/*
 * env.cpp: Multithreading environment 
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

        static void init();
        static int env_get_num_threads();
        static void env_set_offset();
};
int Env::nthreads = 0;
std::vector<uint64_t> Env::start_col;
std::vector<uint64_t> Env::end_col;
std::vector<uint64_t> Env::start_nnz;
std::vector<uint64_t> Env::end_nnz;
std::vector<uint64_t> Env::length_nnz;
std::vector<uint64_t> Env::offset_nnz;


void Env::init() {
    nthreads = env_get_num_threads();
    start_col.resize(nthreads);
    end_col.resize(nthreads);
    start_nnz.resize(nthreads);
    end_nnz.resize(nthreads);
    length_nnz.resize(nthreads);
    offset_nnz.resize(nthreads);
}

int Env::env_get_num_threads() {
    int nthreads_ = 0;
    #pragma omp parallel
    {
        nthreads_ = omp_get_num_threads();
    }
    return(nthreads_);
}    

void Env::env_set_offset() {    
    offset_nnz[0] = 0;
    start_nnz[0] = 0;
    end_nnz[0] = length_nnz[0];
    for(uint32_t i = 1; i < Env::nthreads; i++) {
        start_nnz[i] = end_nnz[i-1];
        end_nnz[i] = start_nnz[i] + length_nnz[i];
        offset_nnz[i] = start_nnz[i];
        //offset_nzz[i] = (length_nnz[-1] + offset_nnz[i-1]);
        
    }
    for(uint32_t i = 0; i < Env::nthreads; i++) {
        printf("%lu %lu %lu %lu %lu %lu\n", start_col[i], end_col[i], start_nnz[i], end_nnz[i], length_nnz[i], offset_nnz[i]);
    }
    //exit(0);
}

#endif 