/*
 * CompressedSpMat.hpp: Compressed Sparse Matrix formats
 * Compressed Sparse Column (CSC)
 * Compressed Sparse Row (CSR)
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@pitt.edu
 */
 
#ifndef CompressedSpMat_HPP
#define CompressedSpMat_HPP

#include <sys/mman.h>
#include <cstring> 
#include "Allocator.hpp"
#include "triple.hpp"





template<typename Weight>
struct CSC {
    public:
        CSC() { nrows = 0, ncols = 0; nnz = 0; IA = nullptr; JA = nullptr; A = nullptr; };
        CSC(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_);
        ~CSC();
        void populate(std::vector<struct Triple<Weight>> &triples);
        void walk();
        uint64_t nnonzeros() { return(nnz); };
        uint64_t nofrows() { return(nrows); };
        uint64_t nofcols() { return(ncols); };
        uint32_t nrows;
        uint32_t ncols;
        uint64_t nnz;
        uint32_t *IA; // Rows
        uint32_t *JA; // Cols
        Weight   *A;  // Vals
        struct Data_Block<uint32_t> *IA_blk;
        struct Data_Block<uint32_t> *JA_blk;
        struct Data_Block<Weight>  *A_blk;
        
};

template<typename Weight>
CSC<Weight>::CSC(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_) {
    nrows = nrows_;
    ncols = ncols_;
    nnz   = nnz_;
    IA = nullptr;
    JA = nullptr;
    A  = nullptr;
    
    //printf("0.%p\n", IA);
    IA_blk = new Data_Block<uint32_t>(&IA, nnz, nnz * sizeof(uint32_t));
    //printf("???\n");
    //printf("??? %d\n", IA[0]);
    
    //printf("3.%p %p\n", IA, IA_blk->ptr);
    //IA_blk->~Data_Block();
    //IA = nullptr;
    //delete IA_blk;
    //IA = nullptr;
    //struct Data_Block<uint32_t> IA_blsk;
    //printf("%p %d\n", IA, IA[0]);
    //printf("%p %p\n", IA, *(IA_blk->ptr));
    
    JA_blk = new Data_Block<uint32_t>(&JA, (ncols + 1), (ncols + 1) * sizeof(uint32_t));
    A_blk  = new Data_Block<Weight>(&A,  nnz, nnz * sizeof(Weight));
    
    //exit(0);
    //IA_blk.ptr = &IA;
    //IA_blk.nitems = nnz;
    //IA_blk.nbytes = nnz * sizeof(uint32_t);
    //IA_blk.allocate();
    
    
    /*
    JA = nullptr;
    JA_blk.ptr = &JA;
    JA_blk.nitems = (ncols + 1);
    JA_blk.nbytes = (ncols + 1) * sizeof(uint32_t);
    
    A = nullptr;
    A_blk.ptr = &A;
    A_blk.nitems = nnz;
    A_blk.nbytes = nnz * sizeof(Weight);
    */
    //printf("%p %p\n", &IA, IA_blk.ptr);
    //exit(0);
    
    /*
    if(ncols and nnz) {
        if((IA = (uint32_t *) mmap(nullptr, nnz * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(IA, 0, nnz * sizeof(uint32_t));
        
        if((JA = (uint32_t *) mmap(nullptr, (ncols + 1) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(JA, 0, (ncols + 1) * sizeof(uint32_t));
        
        if((A = (Weight *) mmap(nullptr, nnz * sizeof(Weight), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(A, 0, nnz * sizeof(Weight));
    }
    */
}

template<typename Weight>
CSC<Weight>::~CSC(){
    //printf("deleting\n");
    delete IA_blk;
    IA = nullptr;
    delete JA_blk;
    JA = nullptr;
    delete  A_blk;
    A  = nullptr;
    
    /*
    if(ncols and nnz) {
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
    */
}

template<typename Weight>
void CSC<Weight>::populate(std::vector<struct Triple<Weight>> &triples) {
    if(ncols and nnz) {
        ColSort<double> f_col;
        std::sort(triples.begin(), triples.end(), f_col);
        
        uint32_t i = 0;
        uint32_t j = 1;
        JA[0] = 0;
        for(auto& triple : triples) {
            while((j - 1) != triple.col) {
                j++;
                JA[j] = JA[j - 1];
            }                  
            A[i] = triple.weight;
            JA[j]++;
            IA[i] = triple.row;
            i++;
        }
        while((j + 1) < ncols) {
            j++;
            JA[j] = JA[j - 1];
        }
    }
}

template<typename Weight>
void CSC<Weight>::walk() {
    for(uint32_t j = 0; j < ncols; j++) {
        printf("j=%d\n", j);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            printf("    i=%d, j=%d, value=%f\n", IA[i], j, A[i]);
        }
        
    }
}

#endif