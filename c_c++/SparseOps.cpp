/*
 * SparseOps.cpp: Sparse Matrix operations
 * Sparse Matrix - Sparse Matrix (SpMM)
 * Sparse Matrix - Dense Vector (SpMV)
 * Element-wise Sparse Matrix - Dense Vector (SpMV_EW)
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
#ifndef SPARSEOPS_CPP
#define SPARSEOPS_CPP


template<typename Weight>
inline uint64_t SpMM_Sym(struct CompressedSpMat<Weight> &A, struct CompressedSpMat<Weight> *B,
                         struct DenseVec<Weight> *x) {
                             auto *B_CSC = B->csc;
    uint32_t *IA_B = B_CSC->IA;
    uint32_t *JA_B = B_CSC->JA;
    Weight   *A_B  = B_CSC->A;
    //uint32_t ncols_B = B_CSC->ncols;
    //uint32_t nrows_B = B_CSC->nrows;                         
                             
                             
                             /*
    uint32_t *IA_A = A_CSC->IA;
    uint32_t *JA_A = A_CSC->JA;
    Weight   *A_A  = A_CSC->A;
    uint32_t ncols_A = A_CSC->ncols;
    uint32_t nrows_A = A_CSC->nrows;
    
    uint32_t *IA_B = B_CSC->IA;
    uint32_t *JA_B = B_CSC->JA;
    Weight   *A_B  = B_CSC->A;
    uint32_t ncols_B = B_CSC->ncols;
    uint32_t nrows_B = B_CSC->nrows;
    
    auto *A_V = spa_DVEC->A;

    uint64_t nnzmax = 0;
    
    if(ncols_A != nrows_B) {
        fprintf(stderr, "Error: SpMM dimensions do not agree A[%d %d] B[%d %d]\n", nrows_A, ncols_A, nrows_B, ncols_B);
        exit(1);
    }
    
    for(uint32_t j = 0; j < ncols_B; j++) {
        for(uint32_t k = JA_B[j]; k < JA_B[j+1]; k++) {
            uint32_t l = IA_B[k];
            for(uint32_t m = JA_A[l]; m < JA_A[l+1]; m++) {
                A_V[IA_A[m]] = 1;
            }
        }

        for(uint32_t i = 0; i < nrows_A; i++) {
            if(A_V[i]) {
                nnzmax++;
                A_V[i] = 0;
            }
        }
    }
    
    return(nnzmax);
    */
    return(0);
}

/*
template<typename Weight>
inline uint64_t SpMM_Sym_CSC(struct CSC<Weight> *A_CSC, struct CSC<Weight> *B_CSC,
                         struct DenseVec<Weight> *spa_DVEC) {
    uint32_t *IA_A = A_CSC->IA;
    uint32_t *JA_A = A_CSC->JA;
    Weight   *A_A  = A_CSC->A;
    uint32_t ncols_A = A_CSC->ncols;
    uint32_t nrows_A = A_CSC->nrows;
    
    uint32_t *IA_B = B_CSC->IA;
    uint32_t *JA_B = B_CSC->JA;
    Weight   *A_B  = B_CSC->A;
    uint32_t ncols_B = B_CSC->ncols;
    uint32_t nrows_B = B_CSC->nrows;
    
    auto *A_V = spa_DVEC->A;

    uint64_t nnzmax = 0;
    
    if(ncols_A != nrows_B) {
        fprintf(stderr, "Error: SpMM dimensions do not agree A[%d %d] B[%d %d]\n", nrows_A, ncols_A, nrows_B, ncols_B);
        exit(1);
    }
    
    for(uint32_t j = 0; j < ncols_B; j++) {
        for(uint32_t k = JA_B[j]; k < JA_B[j+1]; k++) {
            uint32_t l = IA_B[k];
            for(uint32_t m = JA_A[l]; m < JA_A[l+1]; m++) {
                A_V[IA_A[m]] = 1;
            }
        }

        for(uint32_t i = 0; i < nrows_A; i++) {
            if(A_V[i]) {
                nnzmax++;
                A_V[i] = 0;
            }
        }
    }
    return(nnzmax);
}
*/


template<typename Weight>
inline void SpMM_CSC(struct CSC<Weight> *A_CSC, struct CSC<Weight> *B_CSC, struct CSC<Weight> *C_CSC,
                 struct DenseVec<Weight> *spa_DVEC, struct DenseVec<Weight> *x_DVEC) {
    uint32_t *IA_A = A_CSC->IA;
    uint32_t *JA_A = A_CSC->JA;
    Weight   *A_A  = A_CSC->A;
    uint32_t ncols_A = A_CSC->ncols;
    uint32_t nrows_A = A_CSC->nrows;
    
    uint32_t *IA_B = B_CSC->IA;
    uint32_t *JA_B = B_CSC->JA;
    Weight   *A_B  = B_CSC->A;
    uint32_t ncols_B = B_CSC->ncols;
    uint32_t nrows_B = B_CSC->nrows;
    
    uint32_t nrows_C = C_CSC->nrows;
    uint32_t ncols_C = C_CSC->ncols;
    
    auto *A_V = spa_DVEC->A;    
    uint32_t nitems_x = x_DVEC->nitems;
    

    
    if((ncols_A != nrows_B) or (nrows_A != nrows_C) or (ncols_B != ncols_C)) {
        fprintf(stderr, "Error: SpMM dimensions do not agree C[%d %d] != A[%d %d] B[%d %d]\n", nrows_C, ncols_C, nrows_A, ncols_A, nrows_B, ncols_B);
        exit(1);
    }
    
    if(ncols_C != nitems_x) {
        fprintf(stderr, "Error: SpMV_EW dimensions do not agree [%d != %d]\n", ncols_C, nitems_x);
        exit(1);
    }
   
    
    //struct DenseVec<Weight> *spa_DVEC = new struct DenseVec<Weight>(nrows_A);
    
    for(uint32_t j = 0; j < ncols_B; j++) {
        for(uint32_t k = JA_B[j]; k < JA_B[j+1]; k++) {
            uint32_t l = IA_B[k];
            for(uint32_t m = JA_A[l]; m < JA_A[l+1]; m++) {
                A_V[IA_A[m]] += A_B[k] * A_A[m];
            }
        }
        //C_CSC->spapopulate(spa_DVEC, j);
        C_CSC->spapopulate(spa_DVEC, x_DVEC, j);
    } 
    //delete spa_DVEC;
    
}

template<typename Weight>
inline void SpMV_EW_CSC(struct CSC<Weight> *A_CSC, struct DenseVec<Weight> *x_DVEC) {
    Weight YMIN = 0;
    Weight YMAX = 32;
    uint32_t *IA_A = A_CSC->IA;
    uint32_t *JA_A = A_CSC->JA;
    Weight   *A_A  = A_CSC->A;
    uint32_t nrows_A = A_CSC->nrows;
    uint32_t ncols_A = A_CSC->ncols;
    
    Weight   *A_x = x_DVEC->A;
    uint32_t nitems_x = x_DVEC->nitems;
    
    if(ncols_A != nitems_x) {
        fprintf(stderr, "Error: SpMV_EW dimensions do not agree [%d != %d]\n", ncols_A, nitems_x);
        exit(1);
    }
    
    for(uint32_t j = 0; j < ncols_A; j++) {
        for(uint32_t i = JA_A[j]; i < JA_A[j+1]; i++) {
            A_A[i] += A_x[j];
            if(A_A[i] < YMIN) {
                A_A[i] = YMIN;
            }
            else if(A_A[i] > YMAX) {
                A_A[i] = YMAX;
            }
        }
    }
}

#endif