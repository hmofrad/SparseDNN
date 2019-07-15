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
void inferenceReLU(std::vector<struct CompressedSpMat<Weight>*> &layersSpMat, std::vector<struct DenseVec<Weight>*> &biasesDenseVec, struct CompressedSpMat<Weight> *featuresSpMat, Compression_Type compression_type) {    
    auto &W1 = layersSpMat;
    uint32_t maxLayers = W1.size();
    auto &B1 = biasesDenseVec;
    auto *Y0 = featuresSpMat;
    struct CompressedSpMat<Weight> *Y = Y0;
    std::vector<struct DenseVec<Weight>*> spa_VEC;
    for(uint32_t i = 0; i < Env::nthreads; i++) {
        struct DenseVec<Weight> *spa_DVEC = new struct DenseVec<Weight>(Y0->nrows);
        spa_VEC.push_back(spa_DVEC);
    }
    auto &s = spa_VEC;
    uint32_t nrows = 0;
    uint32_t ncols = 0;
    uint32_t nnzcolsmax = 0;
    uint64_t nnzmax = 0;
    struct CompressedSpMat<Weight> *Z;
    if(compression_type == Compression_Type::csc_fmt) {
        
        Z = new struct CompressedSpMat<Weight>(nrows, ncols, nnzcolsmax, nnzmax, compression_type);
        for(uint32_t r = 0; r < maxLayers; r++) {
            auto *W = W1[r];
            auto *W_CSC = W->csc;
            ncols = W_CSC->ncols;
            auto *B = B1[r];
            auto *Y_CSC = Y->csc;
            nrows = Y_CSC->nrows;
            //if(r < 1)
            nnzmax = SpMM_Sym<Weight>(Y_CSC, W_CSC, s);
            //printf("%d:nnzmax=%lu\n", r, nnzmax);
            auto *Z_CSC = Z->csc;
            
            Z_CSC->initialize(nrows, ncols, nnzmax);
            //printf("NOW INIT\n");
            SpMM<Weight>(Y_CSC, W_CSC, Z_CSC, B, s);
            //Z_CSC->walk();
            //printf("DONE INIT\n");
            //Y_CSC->repopulate(Z_CSC);
            
            //Z_CSC->walk();
            //Y_CSC->walk();
            //printf("%d.Y_CSC: nrows=%d ncols=%d nnz=%lu\n", r, Y_CSC->numrows(), Y_CSC->numcols(), Y_CSC->numnonzeros()); 
            //printf("%d.Z_CSC: nrows=%d ncols=%d nnz=%lu\n", r, Z_CSC->numrows(), Z_CSC->numcols(), Z_CSC->numnonzeros()); 
            //exit(0);
        } 
        delete Z;
        
    }

    for(uint32_t i = 0; i < Env::nthreads; i++) {
        delete spa_VEC[i];
    }
    spa_VEC.clear();
    spa_VEC.shrink_to_fit();
}

template<typename Weight>
void validate_prediction(struct CompressedSpMat<Weight> *featuresSpMat, std::vector<uint32_t> trueCategories, Compression_Type compression_type) {
    auto &Y = featuresSpMat;
    uint32_t *JA = nullptr;
    uint32_t *JC = nullptr;
    uint32_t *IA = nullptr;
    Weight   *A = nullptr;
    uint32_t ncols = 0;
    uint32_t nnzcols = 0;
    uint32_t nrows = 0;
    
    if(compression_type == Compression_Type::csc_fmt) {
        auto *Y_CT = Y->csc;
        JA = Y_CT->JA;
        IA = Y_CT->IA;
        A = Y_CT->A;
        ncols = Y_CT->ncols;
        nnzcols = ncols;
        nrows = Y_CT->nrows;
    }

    
    std::vector<Weight> allCategories(nrows);
    for(uint32_t j = 0; j < nnzcols; j++) {
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
