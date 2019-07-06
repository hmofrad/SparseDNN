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

double YMIN = 0;
double YMAX = 32;

template<typename Weight>
inline uint64_t SpMM_Sym(struct CSC<Weight> *A_CSC, struct CSC<Weight> *B_CSC) {
    uint32_t *IA_A = A_CSC->IA;
    uint32_t *JA_A = A_CSC->JA;
    Weight   *A_A  = A_CSC->A;
    uint32_t ncols_A = A_CSC->ncols;
    uint32_t nrows_A = A_CSC->nrows;
    //std::vector<char> &rows_A = A_CSC->validrows;
    
    uint32_t *IA_B = B_CSC->IA;
    uint32_t *JA_B = B_CSC->JA;
    Weight   *A_B  = B_CSC->A;
    uint32_t ncols_B = B_CSC->ncols;
    uint32_t nrows_B = B_CSC->nrows;
/*    
    uint32_t *IA_C = C_CSC->IA;
    uint32_t *JA_C = C_CSC->JA;
    Weight   *A_C  = C_CSC->A;
    uint32_t ncols_C = C_CSC->ncols;
    uint32_t nrows_C = C_CSC->nrows;
    uint32_t idx_C = C_CSC->idx;
    uint32_t nnz_C = C_CSC->nnz;
*/
    uint64_t nnzmax = 0;
    
    /*
    uint32_t i = 0;
    uint32_t j = 0;
    
    uint64_t no = 0;
    for(uint32_t i = 0; i < nrows_A; i++) {
        if(rows_A[i]) {
            for(uint32_t j = 0; j < ncols_B; j++) {
                no++;
                if(JA_B[j+1] - JA_B[j]) {
                    nnzmax++;
                }
            }
        }
    }
    */  
    
    
    //uint32_t nrows;
    
    
    struct DenseVec<Weight> *Vec = new struct DenseVec<Weight>(nrows_A);
    auto *A_V = Vec->A;
    for(uint32_t j = 0; j < ncols_B; j++) {
        for(uint32_t k = JA_B[j]; k < JA_B[j+1]; k++) {
            uint32_t l = IA_B[k];
            //printf("%d %d %d\n", j, k, l);
            //Weight t = 0;      
            //nnzmax += (JA_A[l+1] - JA_A[l]);
            
                
                
            
            for(uint32_t m = JA_A[l]; m < JA_A[l+1]; m++) {
                A_V[IA_A[m]] = 1;
             //   no++;
            }

            
            //for(uint32_t m = JA_A[l]; m < JA_A[l+1]; m++) {
              //  if(A_B[k] * A_A[m])
                        
            //}
                //A_V[IA_A[m]] += A_B[k] * A_A[m];
                
            //}
            //printf("t=%f \n", t);
        }

        for(uint32_t i = 0; i < nrows_A; i++) {
           // no++;
            auto &v = A_V[i];
            if(v) {
                nnzmax++;
                v = 0;
            }
        }
    }
    
    
   // printf("%lu\n", no);
    return(nnzmax);
}



template<typename Weight>
inline void SpMM(struct CSC<Weight> *A_CSC, struct CSC<Weight> *B_CSC, struct CSC<Weight> *C_CSC) {
    uint32_t *IA_A = A_CSC->IA;
    uint32_t *JA_A = A_CSC->JA;
    Weight   *A_A  = A_CSC->A;
    uint32_t ncols_A = A_CSC->ncols;
    uint32_t nrows_A = A_CSC->nrows;
    
    //std::vector<Weight> S_A = A_CSC->colsum;
    /*
    uint32_t *IA_AR = A_CSR->IA;
    uint32_t *JA_AR = A_CSR->JA;
    Weight   *A_AR  = A_CSR->A;
    uint32_t ncols_AR = A_CSR->ncols;
    uint32_t nrows_AR = A_CSR->nrows;
    std::vector<Weight> S_AR = A_CSR->rowsum;
    */
    uint32_t *IA_B = B_CSC->IA;
    uint32_t *JA_B = B_CSC->JA;
    Weight   *A_B  = B_CSC->A;
    uint32_t ncols_B = B_CSC->ncols;
    uint32_t nrows_B = B_CSC->nrows;
    /*
    uint32_t *IA_C = C_CSC->IA;
    uint32_t *JA_C = C_CSC->JA;
    Weight   *A_C  = C_CSC->A;
    uint32_t ncols_C = C_CSC->ncols;
    uint32_t nrows_C = C_CSC->nrows;
    uint32_t idx_C = C_CSC->idx;
    uint32_t nnz_C = C_CSC->nnz;
    */
    uint64_t ops = 0;
    //struct Triple<Weight> triple;
    struct DenseVec<Weight> *Vec = new struct DenseVec<Weight>(nrows_A);
    auto *A_V = Vec->A;
    //printf("AC[%d %d] AR[%d %d] B[%d %d] C[%d %d] %lu/%lu\n", nrows_A, ncols_A, nrows_AR, ncols_AR, nrows_B, ncols_B, nrows_C, ncols_C, nnz_C, idx_C);
    
    //exit(0);
    /*
    for(uint32_t j = 0; j < ncols_B; j++) {
        
        JA_C[j+1] += JA_C[j];
        for(uint32_t k = JA_B[j]; k < JA_B[j+1]; k++) {
            uint32_t l = IA_B[k];
            //printf("%d %d %d\n", j, k, l);
            Weight t = 0;      
            //A_V[IA_A[m]] += A_B[k] * S_A;
            
            
            for(uint32_t m = JA_A[l]; m < JA_A[l+1]; m++) {
                //std::cout << A_B[k] << " "  <<  A_A[m] << std::endl;
                //printf("%f %f %d %f\n", A_B[k], A_A[m], IA_A[m], A_B[k] * A_A[m]);
                //t += A_B[k] * A_A[m]; // IA_A[m] or IA_B[k]
                A_V[IA_A[m]] += A_B[k] * A_A[m];
                
             //   ops++;
            }
            JA[j+1]++;
            
            
            IA[idx] = i;
            A[idx] = v;
            //validrows[i] = 1;
            idx++;
            
            
            
            //printf("t=%f \n", t);
        }
        

        
        
        //C_CSC->populate_spa(Vec, j);
    }
    */
    //exit(0);
    
   
    for(uint32_t j = 0; j < ncols_B; j++) {
        
        
        for(uint32_t k = JA_B[j]; k < JA_B[j+1]; k++) {
            uint32_t l = IA_B[k];
            //printf("%d %d %d\n", j, k, l);
            //Weight t = 0;      
            //A_V[IA_A[m]] += A_B[k] * S_A;
            
            
            for(uint32_t m = JA_A[l]; m < JA_A[l+1]; m++) {
                //std::cout << A_B[k] << " "  <<  A_A[m] << std::endl;
                //printf("%f %f %d %f\n", A_B[k], A_A[m], IA_A[m], A_B[k] * A_A[m]);
                //t += A_B[k] * A_A[m]; // IA_A[m] or IA_B[k]
                A_V[IA_A[m]] += A_B[k] * A_A[m];
                
             //   ops++;
            }
            
   
            
            
            //printf("t=%f \n", t);
        }
        C_CSC->populate_spa(Vec, j);
        
        /*
        for(int v = 0; v < nrows_A; v++) {
            printf("%f ", A_V[v]);
            if(A_V[v]) {
                triple.row = v;
                triple.col = j;
                triple.weight = A_V[v];
                //triples.push_back(triple);
                C_CSC->populate(triple);
            }
        }
        */
        //printf("\n");
    }  
  
    
    
    //printf("OPS=%lu\n", ops);    
}

template<typename Weight>
inline void SpMV_EW(struct CSC<Weight> *A_CSC, struct DenseVec<Weight> *x_VEC) {
    uint32_t *IA_A = A_CSC->IA;
    uint32_t *JA_A = A_CSC->JA;
    Weight   *A_A  = A_CSC->A;
    Weight   *A_x = x_VEC->A;
    uint32_t nrows_A = A_CSC->nrows;
    uint32_t ncols_A = A_CSC->ncols;
    uint32_t nitems_x = x_VEC->nitems;
    
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