/*
 * serial.c: Serial implementation of Sparse Deep Neural Network 
 * for Radix-Net sparse DNN generator
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@pitt.edu
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <cstring> 

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <set>
#include <chrono>

template<typename Weight>
struct CSC {
    public:
        uint64_t nnz;
        uint32_t ncols;
        void *IA;
        void *JA;
        void *A;
        CSC(uint64_t nnz_, uint32_t ncols_) {
            nnz = nnz_;
            ncols = ncols_;
            if(nnz and ncols) {
                if((IA = mmap(nullptr, nnz * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
                    fprintf(stderr, "Error mapping memory\n");
                    exit(1);
                }
                memset(IA, 0, nnz * sizeof(uint32_t));
                
                if((JA = mmap(nullptr, (ncols + 1) * sizeof(uint32_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
                    fprintf(stderr, "Error mapping memory\n");
                    exit(1);
                }
                memset(JA, 0, (ncols + 1) * sizeof(uint32_t));
                
                if((A = mmap(nullptr, nnz * sizeof(Weight), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {    
                    fprintf(stderr, "Error mapping memory\n");
                    exit(1);
                }
                memset(A, 0, nnz * sizeof(Weight));
            }
        }
        ~CSC();   
};


template<typename Weight>
struct Triple {
    uint32_t row;
    uint32_t col;
    Weight weight;
};


//scores = inferenceReLUvec(layers,bias,featureVectors); 

void inferenceReLUvec(std::vector<std::vector<struct Triple<double>>> &layersTriples, std::vector<std::vector<struct Triple<double>>> &biasesTriples, std::vector<struct Triple<double>> &featuresTriples) {
    auto &W = layersTriples;
    auto &B = biasesTriples;
    auto &Y0 = featuresTriples;
    printf("%lu %lu %lu\n", W.size(), B.size(), Y0.size());
    double YMAX = 32;
    auto &Y = Y0;
    //for(uint32_t i = 0; i < W.size(); i++) {
    for(uint32_t i = 0; i < 1; i++) {
        
        printf("%d %lu\n", i, W[i].size());
        
        for(auto &triple: W[i]) {
            printf("row=%d col=%d weight=%f\n", triple.row, triple.col, triple.weight);
            break;
        }
        
        
        printf("%d %lu\n", i, Y.size());
        
        for(auto &triple: Y) {
            printf("row=%d col=%d weight=%f\n", triple.row, triple.col, triple.weight);
            break;
        }
         //Z = Y*W{i};
        
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
    uint32_t NfeatureVectors = Nneurons;
    
    //struct csc featuresCsc;
    
    
    
    
    exit(0);
    
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
    std::vector<std::vector<struct Triple<double>>> layersTriples;
    layersTriples.resize(maxLayers);
    struct Triple<double> layerTriple;  
    
    std::vector<std::vector<struct Triple<double>>> biasesTriples;
    biasesTriples.resize(maxLayers);
    struct Triple<double> biasTriple;  
  
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
        auto &layerTriples = layersTriples[i];
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
        
        auto &biasTriples = biasesTriples[i];
        for(uint32_t j = 0; j < Nneurons; j++) {
            biasTriple.row = 1;
            biasTriple.col = j+1;
            biasTriple.weight = biasValue;
            biasTriples.push_back(biasTriple);
        }
        //printf("INFO: Done  reading the layer file %s\n", layerFile.c_str());
        //printf("INFO: Layer file is %lu x %lu, nnz=%lu\n", nrows, ncols, layersTriples.size());
    }
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
    inferenceReLUvec(layersTriples, biasesTriples, featuresTriples);
    //printf(" %lu %d %d %f\n", biasVector.size(), biasVector[0][1023].row, biasVector[0][1023].col, biasVector[0][1023].weight);
    //scores = inferenceReLUvec(layers,bias,featureVectors); 
    
    return(0);
}
