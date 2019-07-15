/*
 * SparseOps.cpp: Sparse Matrix operations
 * Sparse Matrix - Sparse Matrix (SpMM)
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
#ifndef SPARSEOPS_CPP
#define SPARSEOPS_CPP

#include "Env.hpp"

template<typename Weight>
inline uint64_t SpMM_Sym(struct CSC<Weight> *A_CSC, struct CSC<Weight> *B_CSC,
                        std::vector<struct DenseVec<Weight> *> &s) { 
    uint32_t *A_JA = A_CSC->JA;
    uint32_t *A_IA = A_CSC->IA;      
    Weight   *A_A  = A_CSC->A;
    uint32_t A_nrows = A_CSC->nrows;  
    uint32_t A_ncols = A_CSC->ncols;
    
    uint32_t *B_JA = B_CSC->JA;
    uint32_t *B_IA = B_CSC->IA;      
    Weight   *B_A  = B_CSC->A;
    uint32_t B_nrows = B_CSC->nrows;  
    uint32_t B_ncols = B_CSC->ncols;
    
    uint64_t nnzmax = 0;        
    if(A_ncols != B_nrows) {
        fprintf(stderr, "Error: SpMM dimensions do not agree A[%d %d] B[%d %d]\n", A_nrows, A_ncols, B_nrows, B_ncols);
        exit(1);
    }
    
    Env::env_unset_offset();
    #pragma omp parallel reduction(+:nnzmax)
    {
        long nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        uint32_t length = B_ncols;
        uint32_t chunk = length/nthreads;
        uint32_t start = chunk * tid;
        uint32_t end = start + chunk;
        end = (tid == nthreads - 1) ? length : end;
        uint64_t nnzmax_local = 0;
        auto *s_A = s[tid]->A;
        
        for(uint32_t j = start; j < end; j++) {
            for(uint32_t k = B_JA[j]; k < B_JA[j+1]; k++) {
                uint32_t l = B_IA[k];
                for(uint32_t m = A_JA[l]; m < A_JA[l+1]; m++) {
                    s_A[A_IA[m]] = 1;
                }
            }
            for(uint32_t i = 0; i < A_nrows; i++) {
                if(s_A[i]) {
                    nnzmax_local++;
                    s_A[i] = 0;
                }
            }
        }

        nnzmax += nnzmax_local;
        Env::start_col[tid] = start;
        Env::end_col[tid] = end;
        Env::length_nnz[tid] = nnzmax_local;
    }
    Env::env_set_offset();
    return(nnzmax);
}

template<typename Weight>
inline void SpMM(struct CSC<Weight> *A_CSC, struct CSC<Weight> *B_CSC, struct CSC<Weight> *C_CSC,
                 struct DenseVec<Weight> *x, std::vector<struct DenseVec<Weight> *> &s) {  
    uint32_t *A_JA = A_CSC->JA;
    uint32_t *A_IA = A_CSC->IA;      
    Weight   *A_A  = A_CSC->A;
    uint32_t A_nrows = A_CSC->nrows;  
    uint32_t A_ncols = A_CSC->ncols;    
    
    uint32_t *B_JA = B_CSC->JA;
    uint32_t *B_IA = B_CSC->IA;      
    Weight   *B_A  = B_CSC->A;
    uint32_t B_nrows = B_CSC->nrows;
    uint32_t B_ncols = B_CSC->ncols;

    uint32_t C_nrows = C_CSC->nrows;
    uint32_t C_ncols = C_CSC->ncols;;       
                 
    uint32_t x_nitems = x->nitems;
    
    if((A_ncols != B_nrows) or (A_nrows != C_nrows) or (B_ncols != C_ncols)) {
        fprintf(stderr, "Error: SpMM dimensions do not agree C[%d %d] != A[%d %d] B[%d %d]\n", C_nrows, C_ncols, A_nrows, A_ncols, B_nrows, B_ncols);
        exit(1);
    }
    
    if(C_ncols != x_nitems) {
        fprintf(stderr, "Error: SpMV_EW dimensions do not agree [%d != %d]\n", C_ncols, x_nitems);
        exit(1);
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint32_t start = Env::start_col[tid];
        uint32_t end = Env::end_col[tid];
        auto *s_A = s[tid]->A;

            for(uint32_t j = start; j < end; j++) {
                for(uint32_t k = B_JA[j]; k < B_JA[j+1]; k++) {
                    uint32_t l = B_IA[k];
                    for(uint32_t m = A_JA[l]; m < A_JA[l+1]; m++) {
                        s_A[A_IA[m]] += B_A[k] * A_A[m];
                    }
                }
                C_CSC->spapopulate_t(x, s[tid], j, tid);
            }
            #pragma omp barrier
            C_CSC->postpopulate_t(tid);
            A_CSC->repopulate(C_CSC, tid);
    }
}
#endif