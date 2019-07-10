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
        static std::vector<uint64_t> start;
        static std::vector<uint64_t> end;
        static std::vector<uint64_t> offset;
        static std::vector<uint64_t> length;
        
        static void init();
        static int env_get_num_threads();
        static void env_set_offset();
};
int Env::nthreads = 0;
std::vector<uint64_t> Env::start;
std::vector<uint64_t> Env::end;
std::vector<uint64_t> Env::offset;
std::vector<uint64_t> Env::length;


void Env::init() {
    nthreads = env_get_num_threads();
    start.resize(nthreads);
    end.resize(nthreads);
    offset.resize(nthreads);
    length.resize(nthreads);
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
    offset[0] = 0;
    for(uint32_t i = 1; i < Env::nthreads; i++) {
        offset[i] = (length[i-1] + offset[i-1]);
    }
   // for(uint32_t i = 0; i < Env::nthreads; i++) {
     //   printf("%lu %lu %lu %lu\n", start[i], end[i], length[i], offset[i]);
    //}
}

#endif 