/*
 * InferenceReLU.cpp: Inference Rectified Linear Unit (ReLU) implementation
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
#ifndef INFERENCERELU_CPP
#define INFERENCERELU_CPP

template<typename Weight>
void inferenceReLU(std::vector<struct CompressedSpMat<Weight>*> &layersSpMat, std::vector<struct DenseVec<Weight>*> &biasesDenseVec, struct CompressedSpMat<Weight> &featuresSpMat, Compression_Type compression_type) {    
    auto &W1 = layersSpMat;
    auto &B1 = biasesDenseVec;
    auto &Y0 = featuresSpMat;
    auto &Y = Y0;
    uint32_t maxLayers = W1.size();
    struct CompressedSpMat<Weight> *ZSpMat;
    struct DenseVec<Weight> *spa_DVEC = new struct DenseVec<Weight>(Y0.nrows);
    auto *s = spa_DVEC;
    uint64_t nnzmax;
    if(compression_type == Compression_Type::csc_fmt) {
        for(uint32_t r = 0; r < maxLayers; r++) {
            auto *W = W1[r];
            auto *B = B1[r];
            nnzmax = SpMM_Sym<Weight>(Y, W, s);
            auto *Y_CSC = Y.csc;
            auto *W_CSC = W->csc;
            ZSpMat = new struct CompressedSpMat<Weight>(Y_CSC->nrows, W_CSC->ncols, nnzmax, compression_type);
            auto *Z = ZSpMat;
            SpMM<Weight>(Y, W, Z, B, s);
            auto *Z_CSC = ZSpMat->csc;
            Z_CSC->postpopulate();
            Y_CSC->repopulate(Z_CSC);
            delete ZSpMat;
        }
    }
    else if(compression_type == Compression_Type::dcsc_fmt) 
    {
        ;
      //  exit(0);
    }
    delete spa_DVEC;
}

template<typename Weight>
void validate_prediction(struct CompressedSpMat<Weight> &featuresSpMat, std::vector<uint32_t> trueCategories) {
    auto &Y = featuresSpMat;
    auto *Y_CSC = Y.csc;
    std::vector<Weight> allCategories(Y_CSC->nrows);

    for(uint32_t j = 0; j < Y_CSC->ncols; j++) {
        Weight s = 0;
        for(uint32_t i = Y_CSC->JA[j]; i < Y_CSC->JA[j+1]; i++) {
            allCategories[Y_CSC->IA[i]] += Y_CSC->A[i];
        }
    }
    
    std::vector<int32_t> predictedCategories;
    for(uint32_t i = 0; i < Y_CSC->nrows; i++) {
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
        printf("Challenge PASSED\n");
    }
    else {
        printf("Challenge FAILED\n");
    }
}


/*
template<typename Weight>
void inferenceReLU_DCSC(std::vector<struct CompressedSpMat<double>*> &layersSpMat, std::vector<struct DenseVec<double>*> &biasesDenseVec, struct CompressedSpMat<double> &featuresSpMat) {    
    auto &W1 = layersSpMat;
    auto &B1 = biasesDenseVec;
    auto &Y0 = featuresSpMat;
    auto &Y = Y0;
    //std::vector<struct Triple<double>> triples;
    uint32_t maxLayers = W1.size();
    struct CompressedSpMat<double> *ZSpMat;
    struct DenseVec<Weight> *spa_DVEC = new struct DenseVec<Weight>(Y.csc->nrows);
    uint64_t nnzmax;
    for(uint32_t r = 0; r < maxLayers; r++) {
        auto *W_DCSC = W1[r]->dcsc;
        auto *Y_DCSC = Y.dcsc;
        auto *B = B1[r];
        
        nnzmax = SpMM_Sym_DCSC<double>(Y_DCSC, W_DCSC, spa_DVEC);
        ZSpMat = new struct CompressedSpMat<double>(Y_DCSC->nrows, W_DCSC->ncols, nnzmax, Compression_Type::dcsc_fmt);
        auto *Z_DCSC = ZSpMat->dcsc;
        //SpMM<double>(Y_CSC, W_CSC, Z_CSC, spa_DVEC);
        SpMM_DCSC<double>(Y_DCSC, W_DCSC, Z_DCSC, spa_DVEC, B);
        //SpMM<double>(Y_CSC, W_CSC, Z_CSC);
        Z_DCSC->postpopulate();
        //SpMV_EW_CSC<double> (Z_CSC, B);  
        Y_DCSC->repopulate(Z_DCSC);
        delete ZSpMat;      
        printf("%d.Y_DCSC: nrows=%d ncols=%d nnz=%lu\n", r, Y_DCSC->numrows(), Y_DCSC->numcols(), Y_DCSC->numnonzeros()); 
        
    }
    delete spa_DVEC;
}

template<typename Weight>
void validate_prediction_DCSC(struct CompressedSpMat<Weight> &featuresSpMat, std::vector<uint32_t> trueCategories) {
    ;
}
*/


#endif