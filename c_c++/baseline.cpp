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

#include "triple.hpp"
#include "CompressedSpMat.hpp"




//scores = inferenceReLUvec(layers,bias,featureVectors); 

//void inferenceReLUvec(std::vector<std::vector<struct Triple<double>>> &layersTriples, std::vector<std::vector<struct Triple<double>>> &biasesTriples, std::vector<struct Triple<double>> &featuresTriples) {
template<typename Weight>    
void inferenceReLUvec(std::vector<struct CompressedSpMat<double>*> &layersSpMat, std::vector<struct CompressedSpMat<double>*> &biasesSpMat, struct CompressedSpMat<uint32_t> &featuresSpMat) {    
    auto &W1 = layersSpMat;
    auto &B = biasesSpMat;
    auto &Y0 = featuresSpMat;
    //printf("%lu %lu %lu\n", W.size(), B.size(), Y0.size());
    double YMAX = 32;
    auto &Y = Y0;
    //for(uint32_t i = 0; i < W.size(); i++) {
    for(uint32_t i = 0; i < 1; i++) {
        auto *W = W1[i];
        auto *Y_CSC = Y.csc;
        auto *Y_CSR = Y.csr;
        auto *W_CSC = W->csc;
        auto *W_CSR = W->csr;
        
        Y_CSR->walk();
        printf("\n");
        W_CSR->walk();
        
        
        //Z = Y*W{i};
        
        struct Triple<double> triple;
        std::vector<struct Triple<double>> triples;
        uint64_t nrows = 0; 
        uint64_t ncols = 0;
        for(uint32_t i = 0; i < Y_CSR->nrows; i++) {
            //printf("i=%d\n", i);
            for(uint32_t j = Y_CSR->IA[i]; j < Y_CSR->IA[i+1]; j++) {
                //Y_CSR->JA[j];
                ///Y_CSR->A[j];
                for(uint32_t k = W_CSR->IA[Y_CSR->JA[j]]; k < W_CSR->IA[Y_CSR->JA[j]+1]; k++) {
                    Y_CSR->A[j];
                    W_CSR->JA[k];
                    W_CSR->A[k];
                    triple.row = i;
                    triple.col = W_CSR->JA[k];
                    triple.weight = 0;
                    triples.push_back(triple);
                    ncols++;
                    
                   /// if(W_CSR->JA[k] > ncols)
                      //  ncols = W_CSR->JA[k];
                 //   printf("i=[%d %f] [j=%d %f]\n", i, Y_CSR->A[j], Y_CSR->JA[k], Y_CSR->A[k]);
                }
                break;
            }
        }
        //printf("ncols = %lu %d %d %d\n", ncols, Y_CSR->nrows, W_CSC->ncols, triples.size());
        struct CompressedSpMat<double> ZSpMat(Y_CSR->nrows, W_CSC->ncols, triples.size(), triples, Compression_Type::dual);
        triples.clear();
        triples.shrink_to_fit();
        printf("\n");
        //ZSpMat.csr->walk();
        //exit(0);
        auto &Z = ZSpMat;
        auto *Z_CSR = Z.csr;
        Z_CSR->walk();
        uint64_t l = 1;
        for(uint32_t i = 0; i < Y_CSR->nrows; i++) {
            printf("i=%d\n", i);
            for(uint32_t j = Y_CSR->IA[i]; j < Y_CSR->IA[i+1]; j++) {
                printf("  j=%d\n", Y_CSR->JA[j]);
                for(uint32_t k = W_CSR->IA[Y_CSR->JA[j]], l = Z_CSR->IA[i]; k < W_CSR->IA[Y_CSR->JA[j]+1], l < Z_CSR->IA[i+1]; k++, l++) {
                    //, l = Z_CSR->JA[Z_CSR->IA[i]]
                     //l < Z_CSR->JA[Z_CSR->IA[i+1]];
                    printf("##############   %d %d %d %d\n", i, j, k, l);
                    //Z_CSR->A[l] += 
                    //uint32_t l = Z_CSR->IA[i];
                    Z_CSR->A[l] += Y_CSR->A[j] * W_CSR->A[k];
                }
                
                //Y_CSR->JA[j];
                //Y_CSR->A[j];
               // for(uint32_t k = W_CSR->IA[Y_CSR->JA[j]], k1 = Z_CSR->JA[Z_CSR->IA[i]]; k < W_CSR->IA[Y_CSR->JA[j]+1] && k1 =< Z_CSR->JA[Z_CSR->IA[i]]; k++, k1++) {
                    //Y_CSR->A[j];
                    //W_CSR->JA[k];
                    //W_CSR->A[k];
                    //printf("i=%f, j=%f, %f\n", Y_CSR->A[j] * W_CSR->A[k], Y_CSR->A[j], W_CSR->A[k]);
                    //for(uint32_t k = W_CSR->IA[Y_CSR->JA[j]]; k < W_CSR->IA[Y_CSR->JA[j]+1]; k++) {
                    
                    
                 //   uint32_t ;
                   // Z_CSR->A[k1] += Y_CSR->A[j] * W_CSR->A[k];
                    //Z_CSR->A[Z_CSR->IA[i]] += Y_CSR->A[j] * W_CSR->A[k];

                    
                   /// if(W_CSR->JA[k] > ncols)
                      //  ncols = W_CSR->JA[k];
                 //   printf("i=[%d %f] [j=%d %f]\n", i, Y_CSR->A[j], Y_CSR->JA[k], Y_CSR->A[k]);
               // }
            }
        }
        
        for(uint32_t i = 0; i < Z_CSR->nrows; i++) {
            printf("i=%d\n", i);
            for(uint32_t j = Z_CSR->IA[i]; j < Z_CSR->IA[i+1]; j++) {
                  printf("i=%d, j=%d, %f\n", i, Z_CSR->JA[j], Z_CSR->A[j]);
                
            }
        }
        
        /*
        for(uint32_t j = 0; j < W_CSC->ncols; j++) {
            printf("j=%d\n", j);
            for(uint32_t i = W_CSC->JA[j]; i < W_CSC->JA[j+1]; i++) {
                W_CSC->IA[i];
                W_CSC->A[i];
                //printf("i=%d j=%d v=%f\n", i, Y_CSR->JA[j], Y_CSR->A[j]);
            }
        }
        */
        /*
        for(uint32_t j = 0; j < ncols; j++) {
            printf("j=%d\n", j);
            for(uint32_t k = Y_JA[j]; k < Y_JA[j + 1]; k++) {
                printf("    i=%d, j=%d, value=%f\n", Y_IA[k], j, Y_A[k]);
                kk = 1;
                break;
            }
            if(kk) break;
        }
        */
        
        
  //      for(auto &triple: W[i]) {
//            printf("row=%d col=%d weight=%f\n", triple.row, triple.col, triple.weight);
          //  break;
        //}
        
        
        //printf("Y: nrows=%lu ncols=%lu nnz=%lu %d\n", Y_CSC->nrows, Y_CSC->ncols, Y_CSC->nnz, Y_CSC->JA[100]);
        //printf("W: nrows=%lu ncols=%lu nnz=%lu %d\n", W_CSC->nrows, W_CSC->ncols, W_CSC->nnz, W_CSC->JA[10]);
        
        //for(auto &triple: Y) {
           // printf("row=%d col=%d weight=%f\n", triple.row, triple.col, triple.weight);
          //  break;
        //}
         
        //printf("row=%d col=%d weight=%f\n", Y.);
        
        /*
        uint32_t ncols = Y.ncols;        
        uint32_t *Y_IA = (uint32_t *) Y.IA;
        uint32_t *Y_JA = (uint32_t *) Y.JA;
        double   *Y_A  = (double   *) Y.A;
        
        int kk = 0;
        for(uint32_t j = 0; j < ncols; j++) {
            printf("j=%d\n", j);
            for(uint32_t k = Y_JA[j]; k < Y_JA[j + 1]; k++) {
                printf("    i=%d, j=%d, value=%f\n", Y_IA[k], j, Y_A[k]);
                kk = 1;
                break;
            }
            if(kk) break;
        }
        auto *W_ = W[i];
        printf(">>>>W=%d %lu %d %p\n", i, W_->nnz, W_->ncols, W_->IA);
        //W_->walk();
        
        printf(">>>>W=%d %lu %d\n", i, W[i]->nnz, W[i]->ncols);
        ncols = W[i]->ncols;
        uint32_t *W_IA = (uint32_t *) W_->IA;
        uint32_t *W_JA = (uint32_t *) W_->JA;
        double   *W_A  = (double   *) W_->A;
        for(uint32_t j = 0; j < ncols; j++) {
            printf("j=%d %d\n", j,  W_JA[j + 1] -  W_JA[j]);
            for(uint32_t k = W_JA[j]; k < W_JA[j + 1]; k++) {
                printf("    i=%d, j=%d, value=%f\n", W_IA[k], j, W_A[k]);
                //break;
                kk = 0;
                break;
            }
            if(!kk) break;
        }
        */
        
        
    
        
    }
}    


int main(int argc, char **argv) {
    printf("INFO: Welcome to Sparse Deep Neural Network Serial Implementation\n");
    
    if(argc != 7) {
        fprintf(stderr, "USAGE: %s -n <Nneurons> -l <maxLayers> <path_to_input> <path_to_dnn>\n", argv[0]);
        exit(1);         
    }
    std::vector<double> neuralNetBias = {-0.3,-0.35,-0.4,-0.45};
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

    uint64_t nrows = 0; 
    uint64_t ncols = 0;
    std::vector<struct Triple<uint32_t>> featuresTriples;
    struct Triple<uint32_t> featuresTriple;
    std::string line;
    std::istringstream iss;
    while (std::getline(fin, line)) {
        iss.clear();
        iss.str(line);
        iss >> featuresTriple.row >> featuresTriple.col >> featuresTriple.weight;
        featuresTriples.push_back(featuresTriple);
        if(featuresTriple.row > nrows)
            nrows = featuresTriple.row;
        if(featuresTriple.col > ncols)
            ncols = featuresTriple.col;
    }
    fin.close();
    
    printf("INFO: Done  reading the features file %s\n", featuresFile.c_str());
    printf("INFO: Features file is %lu x %lu, nnz=%lu\n", nrows, ncols, featuresTriples.size());
    
    //uint32_t NfeatureVectors = Nneurons;
    struct CompressedSpMat<uint32_t> featuresSpMat((nrows + 1), (Nneurons + 1), featuresTriples.size(), featuresTriples, Compression_Type::dual);
    featuresTriples.clear();
    featuresTriples.shrink_to_fit();
    //featuresSpMat.csc->walk();
    //printf("\n");
    //featuresSpMat.csr->walk();
    //return(0);
    
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
    
    std::vector<struct Triple<double>> biasTriples;
    struct Triple<double> biasTriple;  
    std::vector<struct CompressedSpMat<double>*> biasesSpMat;

    printf("INFO: Start reading %d layer files\n", maxLayers);
    auto start = std::chrono::high_resolution_clock::now();
    for(uint32_t i = 0; i < maxLayers; i++) {  
        std::string layerFile = ((std::string) argv[6]) + "/neuron" + std::to_string(Nneurons) + "/n" + std::to_string(Nneurons) + "-l" + std::to_string(i+1) + ".tsv";
        
        fin.clear();
        fin.open(layerFile.c_str());
        if(!fin.is_open()) {
            fprintf(stderr, "Error: Opening %s\n", layerFile.c_str());
            exit(1);
        }
        nrows = 0;
        ncols = 0;

        while (std::getline(fin, line)) {
            iss.clear();
            iss.str(line);
            iss >> layerTriple.row >> layerTriple.col >> layerTriple.weight;
            layerTriples.push_back(layerTriple);
            if(layerTriple.row > nrows)
                nrows = layerTriple.row;
            if(layerTriple.col > ncols)
                ncols = layerTriple.col;
        }
        fin.close();
        DNNedges += layerTriples.size();
        
        struct CompressedSpMat<double> *layerSpMat = new struct CompressedSpMat<double>((nrows + 1), (Nneurons + 1), layerTriples.size(), layerTriples, Compression_Type::dual);
        layersSpMat.push_back(layerSpMat);
        layerTriples.clear();
        layerTriples.shrink_to_fit();

        for(uint32_t j = 0; j < Nneurons; j++) {
            biasTriple.row = 1;
            biasTriple.col = j+1;
            biasTriple.weight = biasValue;
            biasTriples.push_back(biasTriple);
        }
        
        struct CompressedSpMat<double> *biasSpMat = new struct CompressedSpMat<double>((nrows + 1), (Nneurons + 1), biasTriples.size(), biasTriples, Compression_Type::dual);
        biasesSpMat.push_back(biasSpMat);
        
        biasTriples.clear();
        biasTriples.shrink_to_fit();
    }    
    auto finish = std::chrono::high_resolution_clock::now();
    printf("INFO: Done  reading %d layer files\n", maxLayers);
    double readLayerTime = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
    double readLayerRate = (double) DNNedges/readLayerTime;
    printf("DNN neurons/layer: %d, layers:%d, edges:%lu\n", Nneurons, maxLayers, DNNedges);
    printf("Read time (sec): %f, read rate (edges/sec): %f\n", readLayerTime, readLayerRate);
    inferenceReLUvec<double>(layersSpMat, biasesSpMat, featuresSpMat);
    for(uint32_t i = 0; i < maxLayers; i++) {  
        delete layersSpMat[i];
        delete biasesSpMat[i];
    }
    return(0);
}
