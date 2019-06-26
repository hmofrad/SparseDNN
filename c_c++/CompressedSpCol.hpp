/*
 * CompressedSpMat.hpp: Compressed Sparse Column (CSC) implementation
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@pitt.edu
 */
 
#ifndef CompressedSpCol_HPP
#define CompressedSpCol_HPP

#include <sys/mman.h>
#include <cstring> 


template<typename Weight>
struct CSC {
    public:
        CSC(uint64_t nnz_,  uint32_t ncols_);
        ~CSC();
        void populate(std::vector<struct Triple<Weight>> triples);
        void walk();
        uint64_t nnz;
        uint32_t ncols;
        void *IA; // Rows
        void *JA; // Cols
        void *A;  // Vals
};

template<typename Weight>
CSC<Weight>::CSC(uint64_t nnz_, uint32_t ncols_) {
    nnz = nnz_;
    ncols = ncols_;
    if(nnz and ncols) {
        if((IA = mmap(nullptr, nnz * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(IA, 0, nnz * sizeof(uint32_t));
        
        if((JA = mmap(nullptr, (ncols + 1) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(JA, 0, (ncols + 1) * sizeof(uint32_t));
        
        if((A = mmap(nullptr, nnz * sizeof(Weight), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(A, 0, nnz * sizeof(Weight));
    }
}

template<typename Weight>
CSC<Weight>::~CSC(){
    if(nnz and ncols) {
        if(munmap(A, nnz * sizeof(uint32_t)) == -1) {
            fprintf(stderr, "Error unmapping memory\n");
            exit(1);
        }
        
        if(munmap(IA, nnz * sizeof(uint32_t)) == -1) {
            fprintf(stderr, "Error unmapping memory\n");
            exit(1);
        }
        
        if(munmap(JA, (ncols + 1) * sizeof(uint32_t)) == -1) {
            fprintf(stderr, "Error unmapping memory\n");
            exit(1);
        }
    }
    printf("deallocate???\n");
}

template<typename Weight>
void CSC<Weight>::populate(std::vector<struct Triple<Weight>> triples) {
    uint32_t *IA = (uint32_t*) this->IA;
    uint32_t *JA = (uint32_t*) this->JA;
    uint32_t *A  = (uint32_t*) this->A;
    
    uint32_t i = 0;
    uint32_t j = 1;
    JA[0] = 0;
    for(auto& triple : triples) {
        while((j - 1) != triple.col) {
            j++;
            JA[j] = JA[j - 1];
        }                  
        A[i] = 1;
        JA[j]++;
        IA[i] = triple.row;
        i++;
    }
    while((j + 1) < ncols) {
        j++;
        JA[j] = JA[j - 1];
    }
}

template<typename Weight>
void CSC<Weight>::walk() {
    uint32_t* A  = (uint32_t*) this->A;
    uint32_t* IA = (uint32_t*) this->IA;
    uint32_t* JA = (uint32_t*) this->JA;
    for(uint32_t j = 0; j < ncols; j++) {
        printf("j=%d\n", j);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            printf("    i=%d, j=%d, value=%d\n", IA[i], j, A[i]);
        }
    }
}

#endif