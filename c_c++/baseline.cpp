/*
 * main.cpp: Driver code for running Sparse Deep Neural Network 
 * for Radix-Net sparse DNN generator
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com
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

#include <omp.h>

#include "Triple.hpp"
#include "DenseVec.hpp"
#include "SparseMat.hpp"
#include "SparseOps.cpp"
#include "InferenceReLU.cpp"
#include "Env.hpp"

//std::vector<

using WGT = double; 
Compression_Type CT = Compression_Type::csc_fmt;
int main(int argc, char **argv) {
    printf("INFO: Welcome to Sparse Deep Neural Network Implementation\n");
    
    if(argc != 7) {
        fprintf(stderr, "USAGE: %s -n <Nneurons> -l <maxLayers> <path_to_input> <path_to_dnn>\n", argv[0]);
        exit(1);         
    }
    std::vector<WGT> neuralNetBias = {-0.3,-0.35,-0.4,-0.45};
    uint32_t Nneurons = atoi(argv[2]);
    std::vector<uint32_t> NneuronsVector = {1024, 4096, 16384, 65536};
    std::ptrdiff_t idxN = std::distance(NneuronsVector.begin(), std::find(NneuronsVector.begin(), NneuronsVector.end(), Nneurons));
    if(idxN >= NneuronsVector.size()) {
        fprintf(stderr, "Invalid number of neurons/layer %d\n", Nneurons);
        exit(1);
    }    
    WGT biasValue = neuralNetBias[idxN];
    
    std::string featuresFile = ((std::string) argv[5]) + "/sparse-images-" + std::to_string(Nneurons) + ".tsv";
    printf("INFO: Start reading the features file %s\n", featuresFile.c_str());
    std::ifstream fin(featuresFile.c_str());
    if(!fin.is_open()) {
        fprintf(stderr, "Error: Opening %s\n", featuresFile.c_str());
        exit(1);
    }
    
    //Env::init();
    
    
    
    //exit(0);
    
    
    uint64_t nrowsFeatures = 0; 
    uint64_t ncolsFeatures = 0;
    std::vector<struct Triple<WGT>> featuresTriples;
    struct Triple<WGT> featuresTriple;
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
    uint64_t NfeatureVectors = nrowsFeatures;
    struct CompressedSpMat<WGT> *featuresSpMat = new struct CompressedSpMat<WGT>((nrowsFeatures + 1), (Nneurons + 1), featuresTriples.size(), featuresTriples, CT);
    //struct CompressedSpMat<WGT> *featuresSpMat1 = new struct CompressedSpMat<WGT>((nrowsFeatures + 1), (Nneurons + 1), featuresTriples.size(), featuresTriples, CT);
    //struct CompressedSpMat<WGT> *layerSpMat = new struct CompressedSpMat<WGT>((Nneurons + 1), (ncols + 1), layerTriples.size(), layerTriples, CT);
    featuresTriples.clear();
    featuresTriples.shrink_to_fit();
    //delete featuresSpMat;
    //exit(0);
    uint32_t maxLayers = atoi(argv[4]);
    std::vector<uint32_t> maxLayersVector = {120, 480, 1920};
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
    
    std::vector<struct Triple<WGT>> layerTriples;
    struct Triple<WGT> layerTriple;  
    std::vector<struct CompressedSpMat<WGT>*> layersSpMat;
    std::vector<struct DenseVec<WGT>*> biasesDenseVec;
    //maxLayers = 4;
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

        uint64_t nrows = 0;
        uint64_t ncols = 0;

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

        struct CompressedSpMat<WGT> *layerSpMat = new struct CompressedSpMat<WGT>((Nneurons + 1), (ncols + 1), layerTriples.size(), layerTriples, CT);
        layersSpMat.push_back(layerSpMat);
        layerTriples.clear();
        layerTriples.shrink_to_fit();
        
        struct DenseVec<WGT> *biaseDenseVec = new struct DenseVec<WGT>((Nneurons + 1));
        auto &bias_A = biaseDenseVec->A;
        for(uint32_t j = 1; j < Nneurons+1; j++) {
            bias_A[j] = biasValue;
        }
        
        biasesDenseVec.push_back(biaseDenseVec);

    } 

    auto finish = std::chrono::high_resolution_clock::now();
    printf("INFO: Done  reading %d layer files\n", maxLayers);
    WGT readLayerTime = (WGT)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
    WGT readLayerRate = (WGT) DNNedges/readLayerTime;
    printf("INFO: DNN neurons/layer: %d, layers:%d, edges:%lu\n", Nneurons, maxLayers, DNNedges);
    printf("INFO: Read time (sec): %f, read rate (edges/sec): %f\n", readLayerTime, readLayerRate);
    
    Env::init();
    
    start = std::chrono::high_resolution_clock::now();
    inferenceReLU<WGT>(layersSpMat, biasesDenseVec, featuresSpMat, CT);
    finish = std::chrono::high_resolution_clock::now();
    WGT challengeRunTime = (WGT)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
    WGT challengeRunRate = NfeatureVectors * (DNNedges/challengeRunTime);
    printf("INFO: Run time (sec): %f, run rate (edges/sec): %f\n", challengeRunTime, challengeRunRate);
    
    validate_prediction<WGT>(featuresSpMat, trueCategories, CT);
    delete featuresSpMat;
    //delete featuresSpMat1;
    for(uint32_t i = 0; i < maxLayers; i++) {  
        delete layersSpMat[i];
        delete biasesDenseVec[i];
    }
    return(0);
}
