/*
 * SparseOps.cpp: Sparse Matrix operations
 * Sparse Matrix - Dense Vector (SpMV)
 * Element-wise Sparse Matrix - Dense Vector (SpMV_EW)
 * Sparse Matrix - Sparse Matrix (SpMM)
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@pitt.edu
 */
 
#ifndef SPARSEOPS_HPP
#define SPARSEOPS_HPP

double YMIN = 0;
double YMAX = 32;

template<typename Weight>
inline void SpMV_EW(struct CSR<Weight> *A_CSR, struct DenseVec<Weight> *x_VEC) {
    uint32_t *IA_A = A_CSR->IA;
    uint32_t *JA_A = A_CSR->JA;
    Weight   *A_A  = A_CSR->A;
    Weight   *A_x = x_VEC->A;
    uint32_t nrows_A = A_CSR->nrows;
    uint32_t ncols_A = A_CSR->ncols;
    uint32_t nitems_x = x_VEC->nitems;
    
    if(ncols_A != nitems_x) {
        fprintf(stderr, "Error: SpMV_EW dimensions do not agree\n");
        exit(1);
    }
    
    for(uint32_t i = 0; i < nrows_A; i++) {
        for(uint32_t j = IA_A[i]; j < IA_A[i+1]; j++) {
            A_A[j] += A_x[JA_A[j]];
            if(A_A[j] < YMIN) {
                A_A[j] = YMIN;
            }
            else if(A_A[j] > YMAX) {
                A_A[j] = YMAX;
            }
        }
    }
}

template<typename Weight>
inline void SpMM(struct CSR<Weight> *A_CSR, struct CSC<Weight> *B_CSC, struct CSR<Weight> *C_CSR) {
    struct Triple<double> triple;
    
    uint32_t *IA_A = A_CSR->IA;
    uint32_t *JA_A = A_CSR->JA;
    Weight   *A_A  = A_CSR->A;
    uint32_t nrows_A = A_CSR->nrows;
    uint32_t ncols_A = A_CSR->ncols;
    
    uint32_t *IA_B = B_CSC->IA;
    uint32_t *JA_B = B_CSC->JA;
    Weight   *A_B  = B_CSC->A;
    uint32_t ncols_B = B_CSC->ncols;
    uint32_t nrows_B = B_CSC->nrows;
    
    //uint32_t *IA_C = C_CSR->IA;
    //uint32_t *JA_C = C_CSR->JA;
    //Weight   *A_C  = C_CSR->A;
    
    if(ncols_A != nrows_B) {
        fprintf(stderr, "Error: SpMM dimensions do not agree\n");
        exit(1);
    }
    
    for(uint32_t i = 0; i < nrows_A; i++) {
            for(uint32_t j = 0; j < ncols_B; j++) {
                uint32_t k = IA_A[i];
                uint32_t l = JA_B[j];                
                double t = 0.0;
                while((k < IA_A[i+1]) and (l < JA_B[j+1])) {
                    if(JA_A[k] == IA_B[l]) {
                        t += (A_A[k] * A_B[l]);
                        k++;
                        l++;
                    }
                    else if(JA_A[k] < IA_B[l]) {
                        k++;
                    }
                    else {
                        l++;
                    }
                }
                if(t != 0) {
                    triple.row = i;
                    triple.col = j;
                    triple.weight = t;
                    //triples.push_back(triple);
                    C_CSR->populate(triple);
                }
            }
            //Z_CSR->populate_spa(triples);
            //triples.clear();
            //triples.shrink_to_fit();
        }
    
}       


template<typename Weight>
inline void SpMM1(struct CSC<Weight> *A_CSC, struct CSC<Weight> *B_CSC, struct CSC<Weight> *C_CSC) {
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
    
    uint32_t *IA_C = C_CSC->IA;
    uint32_t *JA_C = C_CSC->JA;
    Weight   *A_C  = C_CSC->A;
    uint32_t ncols_C = C_CSC->ncols;
    uint32_t nrows_C = C_CSC->nrows;
    
    
    //struct Triple<Weight> triple;
    struct DenseVec<Weight> *Vec = new struct DenseVec<Weight>(nrows_A);
    for(uint32_t j = 0; j < ncols_B; j++) {
        
        auto *A_V = Vec->A;
        for(uint32_t k = JA_B[j]; k < JA_B[j+1]; k++) {
            uint32_t l = IA_B[k];
            //printf("%d %d %d\n", j, k, l);
            Weight t = 0;      
            for(uint32_t m = JA_A[l]; m < JA_A[l+1]; m++) {
                //std::cout << A_B[k] << " "  <<  A_A[m] << std::endl;
                //printf("%f %f %d %f\n", A_B[k], A_A[m], IA_A[m], A_B[k] * A_A[m]);
                //t += A_B[k] * A_A[m]; // IA_A[m] or IA_B[k]
                A_V[IA_A[m]] += A_B[k] * A_A[m];
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
}

template<typename Weight>
inline void SpMV_EW1(struct CSC<Weight> *A_CSC, struct DenseVec<Weight> *x_VEC) {
    uint32_t *IA_A = A_CSC->IA;
    uint32_t *JA_A = A_CSC->JA;
    Weight   *A_A  = A_CSC->A;
    Weight   *A_x = x_VEC->A;
    uint32_t nrows_A = A_CSC->nrows;
    uint32_t ncols_A = A_CSC->ncols;
    uint32_t nitems_x = x_VEC->nitems;
    
    if(ncols_A != nitems_x) {
        fprintf(stderr, "Error: SpMV_EW dimensions do not agree\n");
        exit(1);
    }
    
    for(uint32_t j = 0; j < ncols_A; j++) {
        for(uint32_t i = JA_A[j]; i < JA_A[j+1]; i++) {
            //A_A[i] += A_x[j];
            A_A[i] += -.3;
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