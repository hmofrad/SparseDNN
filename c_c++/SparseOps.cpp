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
                 
          /*       
    uint32_t *A_JA = nullptr;
    //uint32_t *A_JO = nullptr;
    uint32_t *A_JC = nullptr;
    uint32_t *A_JB = nullptr;
    uint32_t *A_JI = nullptr;
    uint32_t *A_IA = nullptr;
    Weight *A_A = nullptr;
    uint32_t A_ncols = 0;
    uint32_t A_nrows = 0; 
    uint32_t A_nnzcols = 0;     

    uint32_t *B_JA = nullptr;
    //uint32_t *B_JO = nullptr;
    uint32_t *B_JC = nullptr;
    uint32_t *B_JB = nullptr;
    uint32_t *B_JI = nullptr;
    uint32_t *B_IA = nullptr;
    Weight *B_A = nullptr;
    uint32_t B_ncols = 0;
    uint32_t B_nrows = 0; 
    uint32_t B_nnzcols = 0;  
    
    uint32_t C_nrows = 0;
    uint32_t C_ncols = 0;   
    */
    /*
    if((A->type == Compression_Type::csc_fmt) and (B->type == Compression_Type::csc_fmt) and (C->type == Compression_Type::csc_fmt)) {
        auto *A_CT = A->csc;
        A_JA = A_CT->JA;
        //A_JO = A_CT->JO;
        A_IA = A_CT->IA;      
        A_A  = A_CT->A;
        A_ncols = A_CT->ncols;
        A_nrows = A_CT->nrows;  
        A_nnzcols = A_ncols;
        
        auto *B_CT = B->csc;
        B_JA = B_CT->JA;
        //B_JO = B_CT->JO;
        B_IA = B_CT->IA;      
        B_A  = B_CT->A;
        B_ncols = B_CT->ncols;
        B_nrows = B_CT->nrows;  
        B_nnzcols = B_ncols;
        
        auto *C_CT = C->csc;
        C_nrows = C_CT->nrows;
        C_ncols = C_CT->ncols;   
    }

    else
    {
        fprintf(stderr, "Error: Compression is not supported\n");
        exit(1);        
    }
 */
    uint32_t x_nitems = x->nitems;
    
    if((A_ncols != B_nrows) or (A_nrows != C_nrows) or (B_ncols != C_ncols)) {
        fprintf(stderr, "Error: SpMM dimensions do not agree C[%d %d] != A[%d %d] B[%d %d]\n", C_nrows, C_ncols, A_nrows, A_ncols, B_nrows, B_ncols);
        exit(1);
    }
    
    if(C_ncols != x_nitems) {
        fprintf(stderr, "Error: SpMV_EW dimensions do not agree [%d != %d]\n", C_ncols, x_nitems);
        exit(1);
    }
    //auto *C_CT = C->csc;
    //C_CT->test();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint32_t start = Env::start_col[tid];
        uint32_t end = Env::end_col[tid];
        auto *s_A = s[tid]->A;

        //if(C->type == Compression_Type::csc_fmt) {
           // auto *C_CT = C->csc;
            for(uint32_t j = start; j < end; j++) {
                for(uint32_t k = B_JA[j]; k < B_JA[j+1]; k++) {
                //for(uint32_t k = B_JA[j] + B_JO[j]; k < B_JA[j+1] + B_JO[j]; k++) {
                    //k += B_JO[j];
                    uint32_t l = B_IA[k];
                    for(uint32_t m = A_JA[l]; m < A_JA[l+1]; m++) {
                    //for(uint32_t m = A_JA[l] + A_JO[l]; m < A_JA[l+1] + A_JO[l]; m++) {
                        //m += A_JO[l];
                        s_A[A_IA[m]] += B_A[k] * A_A[m];
                    }
                }
                C_CSC->spapopulate_t(x, s[tid], j, tid);
            }
            #pragma omp barrier
            C_CSC->postpopulate_t(tid);
            //#pragma omp barrier
            //auto *A_CT = A->csc;
            A_CSC->repopulate(C_CSC, tid);
            //C_CT->postpopulate_t(tid);
            //#pragma omp barrier
            
        //}
        
    }
    /*
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        C->csc->postpopulate_t(tid);
        
        A->csc->repopulate(C->csc, tid);
    }
    */
    
    //printf("xxxxx\n");
    //j = 0;
    //for(uint32_t j = 0; j < C->csc->ncols; j++) {
    //for(uint32_t j = C_CT->JA[; j < C_CT->ncols; j++) {    
      //  printf("%d %d %d\n", j, C->csc->JA[j], C->csc->JA[j+1] - C->csc->JA[j]);
    //}
    //C->csc->walk();
    //exit(0);
    
    /*
    if(C->type == Compression_Type::csc_fmt) {
        auto *C_CT = C->csc;
        C_CT->postpopulate_t();
    }
    else if(C->type == Compression_Type::dcsc_fmt) {
        auto *C_CT = C->dcsc;
        C_CT->postpopulate_t();
    }
    */
}
#endif