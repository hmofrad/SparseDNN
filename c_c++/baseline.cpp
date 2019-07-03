/*
 * baseline.cpp: Baseline implementation of Sparse Deep Neural Network 
 * for Radix-Net sparse DNN generator
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@pitt.edu
 */

#include <stdio.h>
#include <stdlib.h>


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <set>
#include <chrono>

#include "Triple.hpp"
#include "CompressedSpMat.hpp"
#include "DenseVec.hpp"
#include "SparseOps.cpp"



double elapsed_time1;
std::chrono::high_resolution_clock::time_point start1, finish1;

void tic() { 
    start1 = std::chrono::high_resolution_clock::now();
}
void toc(std::string str = " ");
void toc(std::string str) { 
    finish1 = std::chrono::high_resolution_clock::now();
    elapsed_time1 = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish1-start1).count())/1e9;
    printf("elapsed_time - %s =%f\n", str.c_str(), elapsed_time1);
}




//scores = inferenceReLUvec(layers,bias,featureVectors); 

//void inferenceReLUvec(std::vector<std::vector<struct Triple<double>>> &layersTriples, std::vector<std::vector<struct Triple<double>>> &biasesTriples, std::vector<struct Triple<double>> &featuresTriples) {
template<typename Weight>    
//void inferenceReLUvec(std::vector<struct CompressedSpMat<double>*> &layersSpMat, std::vector<struct CompressedSpMat<double>*> &biasesSpMat, struct CompressedSpMat<uint32_t> &featuresSpMat) {    
void inferenceReLUvec(std::vector<struct CompressedSpMat<double>*> &layersSpMat, std::vector<struct DenseVec<double>*> &biasesDenseVec, struct CompressedSpMat<double> &featuresSpMat) {    
    auto &W1 = layersSpMat;
    auto &B1 = biasesDenseVec;
    auto &Y0 = featuresSpMat;
    //printf("%lu %lu %lu\n", W.size(), B.size(), Y0.size());

    auto &Y = Y0;
    //for(uint32_t i = 0; i < W.size(); i++) {
    struct Triple<double> triple;
    std::vector<struct Triple<double>> triples;
    //uint32_t maxLayers = W1.size();
    uint32_t maxLayers = 1;
    
    /*
    for(uint32_t r = 0; r < maxLayers; r++) {
        auto *W = W1[r];
        auto *W_CSR = W->csr;
        auto *Y_CSC = Y.csc;
        auto *B = B1[r];
        
        
        
        struct CompressedSpMat<double> *ZSpMat = new struct CompressedSpMat<double>(W_CSR->nrows, Y_CSC->ncols, triples.size(), triples, Compression_Type::csr_only, &Y_CSC->rownelems);
        auto *Z_CSR = ZSpMat->csr;
        printf("W_CSR: nrows=%d ncols=%d nnz=%lu\n", W_CSR->numrows(), W_CSR->numcols(), Y_CSC->numnonzeros()); 
        printf("Y_CSC: nrows=%d ncols=%d nnz=%lu\n", Y_CSC->numrows(), Y_CSC->numcols(), Y_CSC->numnonzeros()); 
        printf("Z_CSR: nrows=%d ncols=%d nnz=%lu\n", Z_CSR->numrows(), Z_CSR->numcols(), Z_CSR->numnonzeros()); 
        
  
        SpMM<double>(W_CSR, Y_CSC, Z_CSR);
        Z_CSR->postpopulate();
    //printf("################## postpopulate\n");
        
    //printf("################## is done?\n");    

        SpMV_EW<double> (Z_CSR, B);
        
        
        
        
        //printf("################## repopulate\n");
        W_CSR->repopulate(Z_CSR);
        
        
        printf("Y_CSR: nrows=%d ncols=%d nnz=%lu\n", Y_CSC->numrows(), Y_CSC->numcols(), Y_CSC->numnonzeros()); 
        printf("Z_CSR: nrows=%d ncols=%d nnz=%lu\n", Z_CSR->numrows(), Z_CSR->numcols(), Z_CSR->numnonzeros()); 
        
        printf("5. DONE %d\n", r);
        delete ZSpMat;   
        //exit(0);
        
    }
    */
    
    
    for(uint32_t r = 0; r < maxLayers; r++) {
        
        
        
        
        
        auto *W = W1[r];
        //auto *Y_CSC = Y.csc;
        auto *Y_CSR = Y.csr;
        
        auto *W_CSC = W->csc;
        //auto *W_CSR = W->csr;
        auto *B = B1[r];


        struct CompressedSpMat<double> *ZSpMat = new struct CompressedSpMat<double>(Y_CSR->nrows, W_CSC->ncols, triples.size(), triples, Compression_Type::csr_only, &W_CSC->rownelems);
        auto *Z_CSR = ZSpMat->csr;
        
         
        /*
        for(uint32_t i = 0; i < Y_CSR->nrows; i++) {
            for(uint32_t j = 0; j < W_CSC->ncols; j++) {
                uint32_t k = Y_CSR->IA[i];
                uint32_t l = W_CSC->JA[j];                
                double t = 0.0;
                while((k < Y_CSR->IA[i+1]) and (l < W_CSC->JA[j+1])) {
                    if(Y_CSR->JA[k] == W_CSC->IA[l]) {
                        t += (Y_CSR->A[k] * W_CSC->A[l]);
                        k++;
                        l++;
                    }
                    else if(Y_CSR->JA[k] < W_CSC->IA[l]) {
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
                    Z_CSR->populate(triple);
                }
            }
            //Z_CSR->populate_spa(triples);
            //triples.clear();
            //triples.shrink_to_fit();
        }
        */
        SpMM<double>(Y_CSR, W_CSC, Z_CSR);
        Z_CSR->postpopulate();

        SpMV_EW<double> (Z_CSR, B);

        Y_CSR->repopulate(Z_CSR);
        
        
        printf("Y_CSR: nrows=%d ncols=%d nnz=%lu idx=%lu nb=%lu\n", Y_CSR->numrows(), Y_CSR->numcols(), Y_CSR->numnonzeros(), Y_CSR->idx, Y_CSR->nbytes); 
        printf("Z_CSR: nrows=%d ncols=%d nnz=%lu idx=%lu nb=%lu\n", Z_CSR->numrows(), Z_CSR->numcols(), Z_CSR->numnonzeros(), Z_CSR->idx, Z_CSR->nbytes); 
        
        printf("5. DONE %d\n", r);
        delete ZSpMat;   
    }

}    


int main(int argc, char **argv) {
    printf("INFO: Welcome to Sparse Deep Neural Network Serial Implementation\n");
    
    if(argc != 7) {
        fprintf(stderr, "USAGE: %s -n <Nneurons> -l <maxLayers> <path_to_input> <path_to_dnn>\n", argv[0]);
        exit(1);         
    }
    std::vector<double> neuralNetBias = {0, -0.3,-0.35,-0.4,-0.45};
    uint32_t Nneurons = atoi(argv[2]);
    std::vector<uint32_t> NneuronsVector = {3, 1024, 4096, 16384, 65536};
    std::ptrdiff_t idxN = std::distance(NneuronsVector.begin(), std::find(NneuronsVector.begin(), NneuronsVector.end(), Nneurons));
    if(idxN >= NneuronsVector.size()) {
        fprintf(stderr, "Invalid number of neurons/layer %d\n", Nneurons);
        exit(1);
    }    
    double biasValue = neuralNetBias[idxN];
    
    std::string featuresFile = ((std::string) argv[5]) + "/sparse-images-" + std::to_string(Nneurons) + ".tsv";
    printf("INFO: Start reading the features file %s\n", featuresFile.c_str());
    std::ifstream fin(featuresFile.c_str());
    if(!fin.is_open()) {
        fprintf(stderr, "Error: Opening %s\n", featuresFile.c_str());
        exit(1);
    }

    uint64_t nrowsFeatures = 0; 
    uint64_t ncolsFeatures = 0;
    std::vector<struct Triple<double>> featuresTriples;
    struct Triple<double> featuresTriple;
    std::string line;
    std::istringstream iss;
    while (std::getline(fin, line)) {
        iss.clear();
        iss.str(line);
        iss >> featuresTriple.row >> featuresTriple.col >> featuresTriple.weight;
        //iss >> featuresTriple.col >> featuresTriple.row >> featuresTriple.weight;
        featuresTriples.push_back(featuresTriple);
        if(featuresTriple.row > nrowsFeatures)
            nrowsFeatures = featuresTriple.row;
        if(featuresTriple.col > ncolsFeatures)
            ncolsFeatures = featuresTriple.col;
    }
    fin.close();
    
    printf("INFO: Done  reading the features file %s\n", featuresFile.c_str());
    printf("INFO: Features file is %lu x %lu, nnz=%lu\n", nrowsFeatures, ncolsFeatures, featuresTriples.size());
    
    struct CompressedSpMat<double> featuresSpMat((nrowsFeatures + 1), (Nneurons + 1), featuresTriples.size(), featuresTriples, Compression_Type::csr_only);
    featuresTriples.clear();
    featuresTriples.shrink_to_fit();

    
    uint32_t maxLayers = atoi(argv[4]);
    std::vector<uint32_t> maxLayersVector = {1, 120, 480, 1192};
    std::ptrdiff_t idxL = std::distance(maxLayersVector.begin(), std::find(maxLayersVector.begin(), maxLayersVector.end(), maxLayers));
    if(idxL >= maxLayersVector.size()) {
        fprintf(stderr, "Invalid number of layers %d\n", maxLayers);
        exit(1);
    }    
    
    std::string categoryFile = ((std::string) argv[6]) + "/neuron" + std::to_string(Nneurons) + "-l" + std::to_string(maxLayers) + "-categories.tsv";
    printf("INFO: Start reading the category file %s\n", categoryFile.c_str());
    
    fin.clear();
    fin.open(categoryFile.c_str());
    if(!fin.is_open()) {
        fprintf(stderr, "Error: Opening %s\n", categoryFile.c_str());
        exit(1);
    }
    std::vector<uint32_t> trueCategories;
    uint32_t category = 0;
    while (std::getline(fin, line)) {
        iss.clear();
        iss.str(line);
        iss >> category;
        trueCategories.push_back(category);
    }
    fin.close();
    printf("INFO: Done  reading the category file %s\n", categoryFile.c_str());
    uint64_t Ncategories = trueCategories.size();
    printf("INFO: Number of categories %lu\n", Ncategories);

    uint64_t DNNedges = 0;
    
    std::vector<struct Triple<double>> layerTriples;
    struct Triple<double> layerTriple;  
    std::vector<struct CompressedSpMat<double>*> layersSpMat;
    
    //std::vector<struct Triple<double>> biasTriples;
    //struct Triple<double> biasTriple;  
    //std::vector<struct CompressedSpMat<double>*> biasesSpMat;
    std::vector<struct DenseVec<double>*> biasesDenseVec;

    printf("INFO: Start reading %d layer files\n", maxLayers);
    maxLayers = 3;
    auto start = std::chrono::high_resolution_clock::now();
    for(uint32_t i = 0; i < maxLayers; i++) {  
        std::string layerFile = ((std::string) argv[6]) + "/neuron" + std::to_string(Nneurons) + "/n" + std::to_string(Nneurons) + "-l" + std::to_string(i+1) + ".tsv";
        
        fin.clear();
        fin.open(layerFile.c_str());
        if(!fin.is_open()) {
            fprintf(stderr, "Error: Opening %s\n", layerFile.c_str());
            exit(1);
        }

        uint64_t nrows = 0;
        uint64_t ncols = 0;

        while (std::getline(fin, line)) {
            iss.clear();
            iss.str(line);
            iss >> layerTriple.row >> layerTriple.col >> layerTriple.weight;
            //iss >> layerTriple.col >> layerTriple.row >> layerTriple.weight;
            layerTriples.push_back(layerTriple);
            if(layerTriple.row > nrows)
                nrows = layerTriple.row;
            if(layerTriple.col > ncols)
                ncols = layerTriple.col;
        }
        fin.close();
        DNNedges += layerTriples.size();

        struct CompressedSpMat<double> *layerSpMat = new struct CompressedSpMat<double>((Nneurons + 1), (ncols + 1), layerTriples.size(), layerTriples, Compression_Type::csc_only, &featuresSpMat.csr->rowncols);
        //struct CompressedSpMat<double> *layerSpMat = new struct CompressedSpMat<double>((Nneurons + 1), (Nneurons + 1), layerTriples.size(), layerTriples, Compression_Type::csr_only);
        layersSpMat.push_back(layerSpMat);
        layerTriples.clear();
        layerTriples.shrink_to_fit();
        
        //struct DenseVec<double> *biaseDenseVec = new struct DenseVec<double>((ncolsFeatures + 1));
        struct DenseVec<double> *biaseDenseVec = new struct DenseVec<double>((Nneurons + 1));
        auto &bias_A = biaseDenseVec->A;
        //for(uint32_t j = 1; j < ncolsFeatures+1; j++) {
        for(uint32_t j = 1; j < Nneurons+1; j++) {
            bias_A[j] = biasValue;
        }
        
        biasesDenseVec.push_back(biaseDenseVec);

    } 
    
    /*
    struct CompressedSpMat<double> featuresSpMat((Nneurons + 1),(ncolsFeatures + 1), featuresTriples.size(), featuresTriples, Compression_Type::csc_only, &layersSpMat[0]->csr->rowncols);
    featuresTriples.clear();
    featuresTriples.shrink_to_fit();
    */
    
    auto finish = std::chrono::high_resolution_clock::now();
    printf("INFO: Done  reading %d layer files\n", maxLayers);
    double readLayerTime = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
    double readLayerRate = (double) DNNedges/readLayerTime;
    printf("DNN neurons/layer: %d, layers:%d, edges:%lu\n", Nneurons, maxLayers, DNNedges);
    printf("Read time (sec): %f, read rate (edges/sec): %f\n", readLayerTime, readLayerRate);

    start = std::chrono::high_resolution_clock::now();
    inferenceReLUvec<double>(layersSpMat, biasesDenseVec, featuresSpMat);
    finish = std::chrono::high_resolution_clock::now();
    double challengeRunTime = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
    double challengeRunRate = Nneurons * (DNNedges/challengeRunTime);
    
    
    printf("Run time (sec): %f, run rate (edges/sec): %f\n", challengeRunTime, challengeRunRate);
    /*
    std::vector<int32_t> predictedCategories;
    auto &Y = featuresSpMat;
    auto *Y_CSR = Y.csr;
    for(uint32_t i = 0; i < Y_CSR->nrows; i++) {
        double s = 0;
        for(uint32_t j = Y_CSR->IA[i]; j < Y_CSR->IA[i+1]; j++) {
            s += Y_CSR->A[j];
        }
        if( s > 0) {
            predictedCategories.push_back(i);
        }
    }
    */
    /*
    std::vector<int32_t> predictedCategories;
    auto &Y = featuresSpMat;
    auto *Y_CSC = Y.csc;
    for(uint32_t j = 0; j < Y_CSC->ncols; j++) {
        double s = 0;
        for(uint32_t i = Y_CSC->JA[j]; i < Y_CSC->JA[j+1]; i++) {
            s += Y_CSC->A[i];
        }
        if( s > 0) {
            predictedCategories.push_back(j);
        }
    }
    */
    
    /*
    bool tf = true;
    if(Ncategories == predictedCategories.size()) {
        for(int32_t i = 0; i < Ncategories; i++) {
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
    */



 
    
    for(uint32_t i = 0; i < maxLayers; i++) {  
        delete layersSpMat[i];
        delete biasesDenseVec[i];
    }
    return(0);
}
