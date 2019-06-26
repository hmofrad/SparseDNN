/*
 * serial.c: Serial implementation of Sparse Deep Neural Network 
 * for Radix-Net sparse DNN generator
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@pitt.edu
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <set>

template<typename Weight>
struct Triple {
    uint32_t row;
    uint32_t col;
    Weight weight;
};


int main(int argc, char **argv) {
    printf("INFO: Welcome to Sparse Deep Neural Network Serial Implementation\n");
    
    if(argc != 7) {
        fprintf(stderr, "USAGE: %s -n <Nneurons> -l <maxLayers> <path_to_input> <path_to_dnn>\n", argv[0]);
        exit(1);         
    }
    struct timeval t0, t1;

    
    uint32_t Nneurons = atoi(argv[2]);
    std::set<uint32_t> NneuronsSet = {1024, 4096, 16384, 65536};
    if(not (NneuronsSet.count(Nneurons))) {
        fprintf(stderr, "Invalid number of neurons %d\n", Nneurons);
        exit(1);
    }
        
    
    std::string inputFile = ((std::string) argv[5]) + "/sparse-images-" + std::to_string(Nneurons) + ".tsv";
    //inputFile += "sparse-images-" + std::to_string(Nneurons) + ".tsv";
    
    printf("INFO: Start reading the input file %s\n", inputFile.c_str());
    std::ifstream fin(inputFile.c_str());
    if(!fin.is_open()) {
        fprintf(stderr, "Error: Opening %s\n", inputFile.c_str());
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
    std::vector<struct Triple<unsigned char>> inputTriples;
    struct Triple<unsigned char> inputTriple;
    std::string line;
    std::istringstream iss;
    while (std::getline(fin, line)) {
        iss.clear();
        iss.str(line);
        iss >> inputTriple.row >> inputTriple.col >> inputTriple.weight;
        inputTriples.push_back(inputTriple);
        if(inputTriple.row > nrows)
            nrows = inputTriple.row;
        if(inputTriple.col > ncols)
            ncols = inputTriple.col;
        //if(triple.col == 1011)
          //  printf("%d %d %d\n", triple.row, triple.col, triple.weight);
        //exit(0);
    }
    //printf("%d %d %d\n", triples[0].row, triples[0].col, triples[0].weight);
    //printf("size=%lu\n", triples.size());
    fin.close();
    
    printf("INFO: Done  reading the input file %s\n", inputFile.c_str());
    printf("INFO: Input file is %lu x %lu, nnz=%lu\n", nrows, ncols, inputTriples.size());
    uint32_t NfeatureVectors = Nneurons;
    
    uint32_t maxLayers = atoi(argv[4]);
    std::set<uint32_t> maxLayersSet = {120, 480, 1192};
    if(not (maxLayersSet.count(maxLayers))) {
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
        //if(triple.col == 1011)
          //  printf("%d %d %d\n", triple.row, triple.col, triple.weight);
        //exit(0);
    }
    fin.close();
    printf("INFO: Done  reading the category file %s\n", categoryFile.c_str());
    uint64_t Ncategories = trueCategories.size();
    printf("INFO: Number of categories %lu\n", Ncategories);
  
  
    uint64_t DNNedges = 0;
    std::vector<std::vector<struct Triple<double>>> layersTriplesVector;
    layersTriplesVector.resize(maxLayers);
    struct Triple<double> layerTriple;  
  
    gettimeofday(&t0, 0);
    for(uint32_t i = 0; i < maxLayers; i++) {  
        std::string layerFile = ((std::string) argv[6]) + "/neuron" + std::to_string(Nneurons) + "/n" + std::to_string(Nneurons) + "-l" + std::to_string(i+1) + ".tsv";
        //printf("INFO: Start reading the layer file %s\n", layerFile.c_str());
        fin.clear();
        fin.open(layerFile.c_str());
        if(!fin.is_open()) {
            fprintf(stderr, "Error: Opening %s\n", layerFile.c_str());
            exit(1);
        }
        nrows = 0;
        ncols = 0;
        auto& layersTriples = layersTriplesVector[i];
        while (std::getline(fin, line)) {
            iss.clear();
            iss.str(line);
            iss >> layerTriple.row >> layerTriple.col >> layerTriple.weight;
            layersTriples.push_back(layerTriple);
            if(layerTriple.row > nrows)
                nrows = layerTriple.row;
            if(layerTriple.col > ncols)
                ncols = layerTriple.col;
        }
        fin.close();
        DNNedges += layersTriples.size();
        //printf("INFO: Done  reading the layer file %s\n", layerFile.c_str());
        //printf("INFO: Layer file is %lu x %lu, nnz=%lu\n", nrows, ncols, layersTriples.size());
    }
    
    //printf("%lu\n", DNNedges);
    gettimeofday(&t1, 0);
    long readLayerTime = (t1.tv_sec - t0.tv_sec);
    double readLayerRate = DNNedges/readLayerTime;
    
        //disp(['DNN neurons/layer: ' num2str(Nneuron(i)) ', layers: ' num2str(maxLayers(j)) ', edges: ' num2str(DNNedges)]);
    //disp(['Read time (sec): ' num2str(readLayerTime) ', read rate (edges/sec): ' num2str(readLayerRate)]);
    printf("DNN neurons/layer: %d, layers:%d, edges:%lu\n", Nneurons, maxLayers, DNNedges);
    printf("Read time (sec): %f %lu\n", readLayerRate, readLayerTime);
    
//    for(uint32_t i = 0; i < maxLayers; i++) {  
  //  printf("%lu %lu %d %d %f\n", i, layersTriplesVector[i].size(), layersTriplesVector[i][0].row, layersTriplesVector[i][0].col, layersTriplesVector[i][0].weight);
    //}
    
    
  //disp([layerFile num2str(Nneurons(i)) '/n' num2str(Nneurons(i)) '-l' num2str(k) '.tsv']);

    //std::cout <<
    /*
    FILE *file = fopen(inputFile.c_str(),"rb");
    if(!file) {
        fprintf(stderr, "Error on opening %s\n", inputFile.c_str());
        exit(1); 
    }
    
    
    if(fclose(file)) {
        fprintf(stderr, "Error on closing %s\n", inputFile.c_str());
        exit(1);
    }
    */
    return(0);
}
