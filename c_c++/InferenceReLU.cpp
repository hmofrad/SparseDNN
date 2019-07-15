/*
 * InferenceReLU.cpp: Inference Rectified Linear Unit (ReLU) implementation
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
#ifndef INFERENCERELU_CPP
#define INFERENCERELU_CPP

#include "SparseOps.cpp"
#include "Env.hpp"

template<typename Weight>
void inferenceReLU(std::vector<struct CSC<Weight>*> &layersSpMat, std::vector<struct DenseVec<Weight>*> &biasesDenseVec, 
                   struct CSC<Weight> *featuresSpMat, std::vector<struct DenseVec<Weight>*> &spa_VEC) {    
    auto &W0 = layersSpMat;
    uint32_t maxLayers = W0.size();
    auto &B1 = biasesDenseVec;
    auto *Y0 = featuresSpMat;
    auto *Y_CSC = Y0;

    uint32_t nrows = 0;
    uint32_t ncols = 0;
    uint64_t nnzmax = 0;    
    struct CSC<Weight> *Z_CSC = new struct CSC<Weight>(nrows, ncols, nnzmax);
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        for(uint32_t r = 0; r < maxLayers; r++) {
            auto *W_CSC = W0[r];
            auto *B = B1[r];
            auto &s = spa_VEC[tid];
            SpMM_Sym<Weight>(Y_CSC, W_CSC, Z_CSC, s, tid);
            SpMM<Weight>(Y_CSC, W_CSC, Z_CSC, s, B, tid);
        }
    } 
    delete Z_CSC;        
}

template<typename Weight>
void validate_prediction(struct CSC<Weight> *featuresSpMat, std::vector<uint32_t> trueCategories) {
    auto *Y_CSC = featuresSpMat;
    uint32_t *JA = Y_CSC->JA;
    uint32_t *IA = Y_CSC->IA;
    Weight   *A = Y_CSC->A;
    uint32_t ncols = Y_CSC->ncols;
    uint32_t nrows = Y_CSC->nrows;

    std::vector<Weight> allCategories(nrows);
    for(uint32_t j = 0; j < ncols; j++) {
        for(uint32_t i = JA[j]; i < JA[j+1]; i++) {
            allCategories[IA[i]] += A[i];
        }
    }
    
    std::vector<int32_t> predictedCategories;
    for(uint32_t i = 0; i < nrows; i++) {
        if(allCategories[i])
            predictedCategories.push_back(i);
    }

    bool tf = true;
    if(trueCategories.size() == predictedCategories.size()) {        
        for(int32_t i = 0; i < trueCategories.size(); i++) {
            if(predictedCategories[i] != trueCategories[i]) {
                tf = false;
                break;
            }
        }
    } 
    else {
        tf = false;
    }
    
    if(tf) {
        printf("INFO: Challenge PASSED\n");
    }
    else {
        printf("INFO: Challenge FAILED\n");
    }
}

#endif
