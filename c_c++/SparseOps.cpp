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
                         struct DenseVec<Weight> *s) {    
    uint32_t *A_JA = nullptr;
    uint32_t *A_JC = nullptr;
    uint32_t *A_JB = nullptr;
    uint32_t *A_JI = nullptr;
    uint32_t *A_IA = nullptr;
    Weight *A_A = nullptr;
    uint32_t A_ncols = 0;
    uint32_t A_nrows = 0; 
    uint32_t A_nnzcols = 0;     

    uint32_t *B_JA = nullptr;
    uint32_t *B_JC = nullptr;
    uint32_t *B_JB = nullptr;
    uint32_t *B_JI = nullptr;
    uint32_t *B_IA = nullptr;
    Weight *B_A = nullptr;
    uint32_t B_ncols = 0;
    uint32_t B_nrows = 0; 
    uint32_t B_nnzcols = 0;  
    
    
    
    if((A.type == Compression_Type::csc_fmt) and (B->type == Compression_Type::csc_fmt)) {
        auto *A_CT = A.csc;
        A_JA = A_CT->JA;
        A_IA = A_CT->IA;      
        A_A  = A_CT->A;
        A_ncols = A_CT->ncols;
        A_nrows = A_CT->nrows;  
        A_nnzcols = A_ncols;
        
        auto *B_CT = B->csc;
        B_JA = B_CT->JA;
        B_IA = B_CT->IA;      
        B_A  = B_CT->A;
        B_ncols = B_CT->ncols;
        B_nrows = B_CT->nrows;  
        B_nnzcols = B_ncols;
    }
    else if((A.type == Compression_Type::dcsc_fmt) and (B->type == Compression_Type::dcsc_fmt)) {
        auto *A_CT = A.dcsc;
        A_JA = A_CT->JA;
        A_JC = A_CT->JC;
        A_JB = A_CT->JB;
        A_JI = A_CT->JI;
        A_IA = A_CT->IA;      
        A_A  = A_CT->A;
        A_ncols = A_CT->ncols;
        A_nrows = A_CT->nrows;  
        A_nnzcols = A_CT->nnzcols;
        
        auto *B_CT = B->dcsc;
        B_JA = B_CT->JA;
        B_JC = B_CT->JC;
        B_JB = B_CT->JB;
        B_JI = B_CT->JI;
        B_IA = B_CT->IA;      
        B_A  = B_CT->A;
        B_ncols = B_CT->ncols;
        B_nrows = B_CT->nrows;  
        B_nnzcols = B_CT->nnzcols;  
    }     
    else
    {
        fprintf(stderr, "Error: Compression is not supported\n");
        exit(1);        
    }

    auto *s_A = s->A;
    uint64_t nnzmax = 0;        
    
    if(A_ncols != B_nrows) {
        fprintf(stderr, "Error: SpMM dimensions do not agree A[%d %d] B[%d %d]\n", A_nrows, A_ncols, B_nrows, B_ncols);
        exit(1);
    }
    //bool tf = false;
    //if(r == 1) tf = false;
    
    //printf("A: %d/%d %d, B:%d/%d %d \n", A_ncols, A_nnzcols,  A.dcsc->nnz, B_ncols, B_nnzcols, B->dcsc->nnz);
    if((A.type == Compression_Type::csc_fmt) and (B->type == Compression_Type::csc_fmt)) {
        //if(tf)
        //printf("%d %d\n", B_nnzcols, A_nnzcols);
        for(uint32_t j = 0; j < B_nnzcols; j++) {
          //  if(tf)
            //printf("  j=%d sz = %d\n",j, B_JA[j+1] - B_JA[j]);
            for(uint32_t k = B_JA[j]; k < B_JA[j+1]; k++) {
                uint32_t l = B_IA[k];
              //  if(tf)
                //printf("    j=%d l=%d sz = %d idx=%d\n",j, l, A_JA[l+1]-A_JA[l], A_JA[l+1]);
                for(uint32_t m = A_JA[l]; m < A_JA[l+1]; m++) {
                    s_A[A_IA[m]] = 1;
                }
            }
            for(uint32_t i = 0; i < A_nrows; i++) {
                if(s_A[i]) {
                    nnzmax++;
                    s_A[i] = 0;
                }
            }
            //if(tf) {
            //if(j == 1)
                //exit(0);
            //}
        }
    }
    else if((A.type == Compression_Type::dcsc_fmt) and (B->type == Compression_Type::dcsc_fmt)) {
        /*
        if(tf) {
            for(uint32_t j = 0; j < 1; j++) {
                for(uint32_t k = A_JA[j]; k < A_JA[j+1]; k++) {
                    printf("j=%d/%d i=%d/%d v=%f\n", j, A_JC[j], k, A_IA[k], A_A[k]);
                }
            }
        }
        */
        
        //if(tf)
          //printf("%d %d\n", B_nnzcols, A_nnzcols);
        for(uint32_t j = 0; j < B_nnzcols; j++) {
            //if(tf)
            //printf("  j=%d/%d sz = %d\n",j, B_JC[j], B_JA[j+1] - B_JA[j]);
            for(uint32_t k = B_JA[j]; k < B_JA[j+1]; k++) {
                
                if(A_JB[B_IA[k]]) {
                    //if(tf) {
                      //  printf("%d %d %d\n", B_IA[k], A_JB[B_IA[k]], A_JI[B_IA[k]]);
                       // exit(0);
                        
                    //}
                    
                    
                    uint32_t l = A_JI[B_IA[k]];
                    //if(tf)
                    //printf("    j=%d/%d i=%d/%d (%d b=%d i=%d %d)  l=%d sz = %d idx=%d\n",j, B_JC[j], B_IA[k], l, A_JB[B_IA[k]], A_JI[B_IA[k]], A_JC[A_JI[B_IA[k]]], l, A_JA[l+1]-A_JA[l], A_JA[l+1]);
                    //if(l == 0) {
                      //  printf("%d %d %d %d\n", j, l, B_IA[k], A_JA[l+1] - A_JA[l]);
                    //}
                    //printf("xxnnzmax=%d\n",l);
                    for(uint32_t m = A_JA[l]; m < A_JA[l+1]; m++) {
                        //printf("xxnnzmax=%lu %lu\n", m, s->nitems);
                        //A_IA[m];
                        /*
                        if(A_IA[m] > s->nitems){
                            printf("xxnnzmax=%lu %lu\n", nnzmax, s->nitems);
                            break;
                        }
                        */
                        s_A[A_IA[m]] = 1;
                      //;
                    }
                    
                }
            }
            for(uint32_t i = 0; i < A_nrows; i++) {
                if(s_A[i]) {
                    nnzmax++;
                    s_A[i] = 0;
                }
            }
            //if(tf) {
            //if(B_JC[j] == 1)
              //  exit(0);
            //}
        }
    }
    
    
    
    printf("nnzmax=%lu %lu\n", nnzmax, s->nitems);
    //exit(0);
    
    
    return(nnzmax);
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
inline void SpMM(struct CompressedSpMat<Weight> &A, struct CompressedSpMat<Weight> *B, struct CompressedSpMat<Weight> *C,
                 struct DenseVec<Weight> *x, struct DenseVec<Weight> *s) {
    uint32_t *A_JA = nullptr;
    uint32_t *A_JC = nullptr;
    uint32_t *A_JB = nullptr;
    uint32_t *A_JI = nullptr;
    uint32_t *A_IA = nullptr;
    Weight *A_A = nullptr;
    uint32_t A_ncols = 0;
    uint32_t A_nrows = 0; 
    uint32_t A_nnzcols = 0;     

    uint32_t *B_JA = nullptr;
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
    
    if((A.type == Compression_Type::csc_fmt) and (B->type == Compression_Type::csc_fmt) and (C->type == Compression_Type::csc_fmt)) {
        auto *A_CT = A.csc;
        A_JA = A_CT->JA;
        A_IA = A_CT->IA;      
        A_A  = A_CT->A;
        A_ncols = A_CT->ncols;
        A_nrows = A_CT->nrows;  
        A_nnzcols = A_ncols;
        
        auto *B_CT = B->csc;
        B_JA = B_CT->JA;
        B_IA = B_CT->IA;      
        B_A  = B_CT->A;
        B_ncols = B_CT->ncols;
        B_nrows = B_CT->nrows;  
        B_nnzcols = B_ncols;
        
        auto *C_CT = C->csc;
        C_nrows = C_CT->nrows;
        C_ncols = C_CT->ncols;   
    }
    else if((A.type == Compression_Type::dcsc_fmt) and (B->type == Compression_Type::dcsc_fmt) and (C->type == Compression_Type::dcsc_fmt)) {
        auto *A_CT = A.dcsc;
        A_JA = A_CT->JA;
        A_JC = A_CT->JC;
        A_JB = A_CT->JB;
        A_JI = A_CT->JI;
        A_IA = A_CT->IA;      
        A_A  = A_CT->A;
        A_ncols = A_CT->ncols;
        A_nrows = A_CT->nrows;  
        A_nnzcols = A_CT->nnzcols;
        
        auto *B_CT = B->dcsc;
        B_JA = B_CT->JA;
        B_JC = B_CT->JC;
        B_JB = B_CT->JB;
        B_JI = B_CT->JI;
        B_IA = B_CT->IA;      
        B_A  = B_CT->A;
        B_ncols = B_CT->ncols;
        B_nrows = B_CT->nrows;  
        B_nnzcols = B_CT->nnzcols;  
        
        auto *C_CT = C->dcsc;
        C_nrows = C_CT->nrows;
        C_ncols = C_CT->ncols;   
    }     
    else
    {
        fprintf(stderr, "Error: Compression is not supported\n");
        exit(1);        
    }
 
    uint32_t x_nitems = x->nitems;
    auto *s_A = s->A;    
    
    if((A_ncols != B_nrows) or (A_nrows != C_nrows) or (B_ncols != C_ncols)) {
        fprintf(stderr, "Error: SpMM dimensions do not agree C[%d %d] != A[%d %d] B[%d %d]\n", C_nrows, C_ncols, A_nrows, A_ncols, B_nrows, B_ncols);
        exit(1);
    }
    
    if(C_ncols != x_nitems) {
        fprintf(stderr, "Error: SpMV_EW dimensions do not agree [%d != %d]\n", C_ncols, x_nitems);
        exit(1);
    }
    
    if(C->type == Compression_Type::csc_fmt) {
        auto *C_CT = C->csc;
        for(uint32_t j = 0; j < B_nnzcols; j++) {
            for(uint32_t k = B_JA[j]; k < B_JA[j+1]; k++) {
                uint32_t l = B_IA[k];
                for(uint32_t m = A_JA[l]; m < A_JA[l+1]; m++) {
                    s_A[A_IA[m]] += B_A[k] * A_A[m];
                }
            }
            C_CT->spapopulate(x, s, j);
        } 
    }
    else if (C->type == Compression_Type::dcsc_fmt) {
        auto *C_CT = C->dcsc;
        for(uint32_t j = 0; j < B_nnzcols; j++) {
           // if(j == 61)
                //printf("j=%d jc=%d sz=%d %d %d\n", j, B_JC[j], B_JA[j+1] - B_JA[j], A_ncols, A_nnzcols);
            uint32_t ll = 0;
            for(uint32_t k = B_JA[j]; k < B_JA[j+1]; k++) {
                if(A_JB[B_IA[k]]) {
                    uint32_t l = A_JI[B_IA[k]];
                    //printf("%d == %d\n", B_IA[k], A_JC[l]);
                    //if(l == 0)
                      //  printf("AAAAH\n");
                    //ll++;
                    //if(l >= A_nnzcols) {
                        //printf("jb=%d: lb=%d ja=%d\n", j, l, A_AU[l]);
                        //exit(0);
                    //}
                        
                    //if(j == 61)
                      //  printf("  l=%d/%d\n", l, ll);
                    for(uint32_t m = A_JA[l]; m < A_JA[l+1]; m++) {
                        //A_A[m];
                        //B_A[k];
                        //A_IA[m];
                        //if(j == 61)
                        //    printf("%d\n", A_IA[m]);
                        //if(A_IA[m] >= s->nitems) break;
                        //s_A[A_IA[m]] = 1;
                        
                        s_A[A_IA[m]] += B_A[k] * A_A[m];
                    }
                }
            }
            C_CT->spapopulate(x, s, j, B_JC[j]);
        } 
    }

}


/*
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
*/

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