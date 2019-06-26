/*
 * serial.c: Serial implementation of Sparse Deep Neural Network 
 * for Radix-Net sparse DNN generator
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@pitt.edu
 */

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <sstream>

struct triple {
    unsigned int row;
    unsigned int col;
    unsigned char val;
};

int main(int argc, char **argv) {
    printf("Welcome to Sparse Deep Neural Network Implementation\n");
    
    if(argc != 7) {
        fprintf(stderr, "USAGE: %s -n <Nneuron> -m <maxLayers> <path_to_input> <path_to_dnn>\n", argv[0]);
        exit(1);         
    }
    
    unsigned int Nneuron = atoi(argv[2]);
    std::string inputFile = ((std::string) argv[5]) + "sparse-images-" + std::to_string(Nneuron) + ".tsv";
    //inputFile += "sparse-images-" + std::to_string(Nneuron) + ".tsv";
    std::ifstream fin(inputFile.c_str());
    if(!fin.is_open()) {
        fprintf(stderr, "Error on opening %s\n", inputFile.c_str());
        exit(1);
    }
    
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
