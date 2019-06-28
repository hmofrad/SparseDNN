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
#include "CompressedSpCol.hpp"




//scores = inferenceReLUvec(layers,bias,featureVectors); 

//void inferenceReLUvec(std::vector<std::vector<struct Triple<double>>> &layersTriples, std::vector<std::vector<struct Triple<double>>> &biasesTriples, std::vector<struct Triple<double>> &featuresTriples) {
template<typename Weight>    
void inferenceReLUvec(std::vector<struct CSC<double>*> &layersCSC, std::vector<struct CSC<double>*> &biasesCSC, struct CSC<double> &featuresCSC) {    
    auto &W1 = layersCSC;
    auto &B = biasesCSC;
    auto &Y0 = featuresCSC;
    //printf("%lu %lu %lu\n", W.size(), B.size(), Y0.size());
    double YMAX = 32;
    auto &Y = Y0;
    //for(uint32_t i = 0; i < W.size(); i++) {
    for(uint32_t i = 0; i < 1; i++) {
        auto *W = W1[i];
        
        
  //      for(auto &triple: W[i]) {
//            printf("row=%d col=%d weight=%f\n", triple.row, triple.col, triple.weight);
          //  break;
        //}
        
        //Z = Y*W{i};
        printf("Y: nrows=%d ncols=%d nnz=%lu\n", Y.nrows, Y.ncols, Y.nnz);
        printf("W: nrows=%d ncols=%d nnz=%lu\n", W->nrows, W->ncols, W->nnz);
        
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
    std::vector<uint32_t> NneuronsVector = {1024, 4096, 16384, 65536};
    std::ptrdiff_t idxN = std::distance(NneuronsVector.begin(), std::find(NneuronsVector.begin(), NneuronsVector.end(), Nneurons));
    if(idxN >= NneuronsVector.size()) {
        fprintf(stderr, "Invalid number of neurons/layer %d\n", Nneurons);
        exit(1);
    }    
    double biasValue = neuralNetBias[idxN];
    
    //std::vector<uint32_t> NneuronsVector = {1024, 4096, 16384, 65536};
    //if(not (NneuronsSet.count(Nneurons))) {
      ///  fprintf(stderr, "Invalid number of neurons %d\n", Nneurons);
        //exit(1);
    //}
    

    
    //
    
    std::string featuresFile = ((std::string) argv[5]) + "/sparse-images-" + std::to_string(Nneurons) + ".tsv";
    printf("INFO: Start reading the features file %s\n", featuresFile.c_str());
    std::ifstream fin(featuresFile.c_str());
    if(!fin.is_open()) {
        fprintf(stderr, "Error: Opening %s\n", featuresFile.c_str());
        exit(1);
    }
    
    
    /*
    // Obtain filesize
    uint64_t fileSize = 0;
    fin.seekg (0, std::ios_base::end);
    fileSize = (uint64_t) fin.tellg();
    fin.clear();
    fin.seekg(0, std::ios_base::beg);
    */
    
    //printf("fileSize = %lu\n", fileSize);
    uint64_t nrows = 0; 
    uint64_t ncols = 0;
    std::vector<struct Triple<double>> featuresTriples;
    struct Triple<double> featuresTriple;
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
    ColSort<double> f_col;
    if(featuresTriples.size()) {
        std::sort(featuresTriples.begin(), featuresTriples.end(), f_col);
    }
    
    
    uint32_t NfeatureVectors = Nneurons;
    
    struct CSC<double> featuresCSC(nrows + 1, Nneurons + 1, featuresTriples.size()); // Pad the input
    featuresCSC.populate(featuresTriples);
    //featuresCSC.walk();
    //return(0);
    
    uint32_t maxLayers = atoi(argv[4]);
    std::vector<uint32_t> maxLayersVector = {120, 480, 1192};
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
    //std::vector<std::vector<struct Triple<double>>> layersTriples;
    //layersTriples.resize(maxLayers);
    std::vector<struct Triple<double>> layerTriples;
    
    struct Triple<double> layerTriple;  
    
    //std::vector<std::vector<struct Triple<double>>> biasesTriples;
    //biasesTriples.resize(maxLayers);
    std::vector<struct Triple<double>> biasTriples;
    
    struct Triple<double> biasTriple;  
    
    std::vector<struct CSC<double>*> layersCSC;
    std::vector<struct CSC<double>*> biasesCSC;
  
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
        //auto &layerTriples = layersTriples[i];
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
        
        
        //printf("INFO: Done  reading the layer file %s\n", layerFile.c_str());
        //printf("INFO: Layer file is %lu x %lu, nnz=%lu\n", nrows, ncols, layersTriples.size());
        
        if(layerTriples.size()) {
            std::sort(layerTriples.begin(), layerTriples.end(), f_col);
        }
        struct CSC<double> *layerCSC = new struct CSC<double>(nrows + 1, Nneurons + 1, layerTriples.size());
        //layerCSC = new CSC(layerTriples.size(), Nneurons);
        //layerCSC->populate(layerTriples);
        //layerCSC = new struct CSC<double>(layerTriples.size(), Nneurons);
        layerCSC->populate(layerTriples);
        //layerCSC.walk();
        layersCSC.push_back(layerCSC);
        //layersCSC[i] = layerCSC;
        layerTriples.clear();
        layerTriples.shrink_to_fit();
        
        //auto &biasTriples = biasesTriples[i];
        for(uint32_t j = 0; j < Nneurons; j++) {
            biasTriple.row = 1;
            biasTriple.col = j+1;
            biasTriple.weight = biasValue;
            biasTriples.push_back(biasTriple);
        }
        
        if(biasTriples.size()) {
            std::sort(biasTriples.begin(), biasTriples.end(), f_col);
        }
        struct CSC<double> *biasCSC = new struct CSC<double>(nrows + 1, Nneurons + 1, biasTriples.size());
        biasCSC->populate(biasTriples);
        //biasCSC.walk();
        biasesCSC.push_back(biasCSC);
        biasTriples.clear();
        biasTriples.shrink_to_fit();
    }
    
    /*
    
    //layersCSC.resize(maxLayers);
    for(uint32_t i = 0; i < maxLayers; i++) {
        auto &layerTriples = layersTriples[i];
        //auto &layerCSC = layersCSC[i];
        if(layerTriples.size()) {
            std::sort(layerTriples.begin(), layerTriples.end(), f_col);
        }
        struct CSC<double> layerCSC(layerTriples.size(), Nneurons);
        layerCSC.populate(layerTriples);
        //layerCSC.walk();
        layersCSC.push_back(layerCSC);
        //layerCSC(layerTriples.size(), Nneurons);
        //layerCSC.populate(layersTriples);
    }
    */
    
        
    //for(uint32_t i = 0; i < maxLayers; i++) {
    //    printf("%d %lu %d %lu\n", i, layersCSC[i].nnz, layersCSC[i].ncols, layersTriples[i].size());
    //}        
    
    auto finish = std::chrono::high_resolution_clock::now();
    printf("INFO: Done  reading %d layer files\n", maxLayers);
    double readLayerTime = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
    //std::cout << std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count() << "ns\n";
    //printf("%f\n", readLayerTime);
    //gettimeofday(&t1, 0);
    //long readLayerTime = (t1.tv_sec - t0.tv_sec);
    double readLayerRate = (double) DNNedges/readLayerTime;
    printf("DNN neurons/layer: %d, layers:%d, edges:%lu\n", Nneurons, maxLayers, DNNedges);
    printf("Read time (sec): %f, read rate (edges/sec): %f\n", readLayerTime, readLayerRate);
    //inferenceReLUvec(layersTriples, biasesTriples, featuresTriples);
    //printf(">>>>1111 W=%d %lu %d %lu\n", 1, layersCSC[0]->nnz, layersCSC[0]->ncols, layersCSC.size());
    //layersCSC[0]->walk();
    //printf(">>>>\n");
    inferenceReLUvec<double>(layersCSC, biasesCSC, featuresCSC);
    //printf(" %lu %d %d %f\n", biasVector.size(), biasVector[0][1023].row, biasVector[0][1023].col, biasVector[0][1023].weight);
    //scores = inferenceReLUvec(layers,bias,featureVectors); 
    
    for(uint32_t i = 0; i < maxLayers; i++) {  
        delete layersCSC[i];
        delete biasesCSC[i];
    }
    
    return(0);
}
