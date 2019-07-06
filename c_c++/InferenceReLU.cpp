/*
 * InferenceReLU.cpp: Inference Rectified Linear Unit (ReLU) implementation
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
#ifndef INFERENCERELU_CPP
#define INFERENCERELU_CPP

template<typename Weight>
void inferenceReLU(std::vector<struct CompressedSpMat<double>*> &layersSpMat, std::vector<struct DenseVec<double>*> &biasesDenseVec, struct CompressedSpMat<double> &featuresSpMat) {    
    auto &W1 = layersSpMat;
    auto &B1 = biasesDenseVec;
    auto &Y0 = featuresSpMat;
    auto &Y = Y0;
    std::vector<struct Triple<double>> triples;
    uint32_t maxLayers = W1.size();
    struct CompressedSpMat<double> *ZSpMat;
    struct DenseVec<Weight> *spa_DVEC = new struct DenseVec<Weight>(Y.csc->nrows);
    for(uint32_t r = 0; r < maxLayers; r++) {
        auto *W_CSC = W1[r]->csc;
        auto *Y_CSC = Y.csc;
        auto *B = B1[r];
        uint64_t nnzmax = SpMM_Sym<double>(Y_CSC, W_CSC);
        //nnzmax = Y_CSC->nrows * W_CSC->ncols;
        //struct CompressedSpMat<double> *
        ZSpMat = new struct CompressedSpMat<double>(Y_CSC->nrows, W_CSC->ncols, nnzmax, triples, Compression_Type::csc_fmt);
        auto *Z_CSC = ZSpMat->csc;
        //SpMM<double>(Y_CSC, W_CSC, Z_CSC, spa_DVEC);
        SpMM<double>(Y_CSC, W_CSC, Z_CSC, spa_DVEC, B);
        //SpMM<double>(Y_CSC, W_CSC, Z_CSC);
        Z_CSC->postpopulate();
        //SpMV_EW<double> (Z_CSC, B);  
        Y_CSC->repopulate(Z_CSC);
        delete ZSpMat;      
        printf("%d.Y_CSC: nrows=%d ncols=%d nnz=%lu\n", r, Y_CSC->numrows(), Y_CSC->numcols(), Y_CSC->numnonzeros()); 
    }
    delete spa_DVEC;
}

#endif