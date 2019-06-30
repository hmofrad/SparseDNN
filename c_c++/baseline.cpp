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
    for(uint32_t r = 0; r < 1; r++) {
        auto *W = W1[r];
        auto *Y_CSC = Y.csc;
        auto *Y_CSR = Y.csr;
        auto *W_CSC = W->csc;
        auto *W_CSR = W->csr;
        
        
        /*
        Y_CSR->walk();
        printf("\n");
        W_CSC->walk();
        //exit(0);
        
        
        
        //W_CSC->walk();
        std::vector<uint32_t> A_rows;
        for(uint32_t i = 0; i < Y_CSR->nrows; i++) {
            if(Y_CSR->IA[i+1] - Y_CSR->IA[i]) {
                A_rows.push_back(i);
                //printf("%d\n", Y_CSR->IA[i+1] - Y_CSR->IA[i]);
            }
        }
        
        std::vector<uint32_t> B_cols;
        for(uint32_t j = 0; j < W_CSC->ncols; j++) {
            if(W_CSC->JA[j+1] - W_CSC->JA[j]) {
                B_cols.push_back(j);
            }
            //else 
              //  printf("%d\n", j);
        }
        
        //for(auto i: B_cols) printf("%d\n", i);
        //printf("\n");
        printf("Y: nrows=%lu ncols=%lu nnz=%lu\n", Y_CSC->nrows, Y_CSC->ncols, Y_CSC->nnz);
        printf("W: nrows=%lu ncols=%lu nnz=%lu\n", W_CSC->nrows, W_CSC->ncols, W_CSC->nnz);
        //exit(0);
        */
        
        
        //Z = Y*W{i};
        
        struct Triple<double> triple;
        std::vector<struct Triple<double>> triples;
        //std::vector<std::vector<struct Triple<double>>> triples;
        //triples.resize(Y_CSC->nrows);
        uint64_t nrows = 0; 
        uint64_t ncols = 0;
        /*
        for(uint32_t i = 0; i < Y_CSR->nrows; i++) {
            printf("i=%d\n", i);
            for(uint32_t j = Y_CSR->IA[i]; j < Y_CSR->IA[i+1]; j++) {
                printf("  j=%d\n", j);
                for(uint32_t k = W_CSR->IA[Y_CSR->JA[j]]; k < W_CSR->IA[Y_CSR->JA[j]+1]; k++) {
                    printf("    k=%d\n", k);
                    printf("        A[%d %d]=[%d] B[%d %d]=[%f] C[%d %d]\n", i, Y_CSR->JA[j], Y_CSR->A[j], Y_CSR->JA[j], W_CSR->JA[k], W_CSR->A[k], i, W_CSR->JA[k]);
                }
            }
        }
        */
        
        for(int i = 0; i < Y_CSR->rowncols.size(); i++) {
            for(int j = 0; j < W_CSC->colnrows.size(); j++) {
                if(Y_CSR->rowncols[i] and W_CSC->colnrows[j])
                    printf("%d %d\n", i, j);
            }
        }
         
         
         exit(0);

        
        for(uint32_t i = 0; i < Y_CSR->nrows; i++) {
                printf("i=%d/sz=%d\n", i, Y_CSR->IA[i+1] - Y_CSR->IA[i]);
        for(uint32_t j = 0; j < W_CSC->ncols; j++) {
            printf("  j=%d/sz=%d\n", j, W_CSC->JA[j+1] - W_CSC->JA[j]);
                //uint32_t i = 1, j = 2;
                uint32_t k = Y_CSR->IA[i], l = W_CSC->JA[j];                
                double t = 0.0;
                while((k < Y_CSR->IA[i+1]) and (l < W_CSC->JA[j+1])) {
                    //printf("  %d < %d and %d < %d\n", k, Y_CSR->IA[i+1], l, W_CSC->JA[j+1]);
                    if(Y_CSR->JA[k] == W_CSC->IA[l]) {
                        t += (Y_CSR->A[k] * W_CSC->A[l]);
                        //printf("    k=%d l = %d --> %d %f\n", Y_CSR->JA[k], W_CSC->IA[l], Y_CSR->A[k], W_CSC->A[l]);
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
                //printf("    %d %d %d\n", i,j, t);
                std::cout << "    " << i << " " << j << " " <<  t << std::endl;
                if(t > 0) {
                    triple.row = i;
                    triple.col = i;
                    triple.weight = t;
                    triples.push_back(triple);
                    printf("<%d %d %f>\n", triple.row, triple.col, triple.weight);
                }
                
            }
        }
        
        for(auto &tt: triples) {
        //    for(auto tt: t) {
                printf("%d %d %f\n", tt.row, tt.col, tt.weight);
            }                
        //}
        
        printf("%d %d %lu\n", Y_CSR->nrows, W_CSC->ncols, triples.size());
        struct CompressedSpMat<double> ZSpMat(Y_CSR->nrows, W_CSC->ncols, triples.size(), triples, Compression_Type::dual);
        triples.clear();
        triples.shrink_to_fit();
        
        ZSpMat.csc->walk();
        printf("\n");
        ZSpMat.csr->walk();
        
                /*
                for(uint32_t k = Y_CSR->IA[i]; k < Y_CSR->IA[i+1]; k++) {
                    printf("    k=%d\n", Y_CSR->JA[k]);
                    for(uint32_t l = W_CSC->JA[j]; l < W_CSC->JA[j+1]; l++) {
                            if(W_CSC->IA[k] == Y_CSR->JA[l])
                                printf("      l=%d \n", W_CSC->IA[l]);
                            
                    }
                }
                */
          //  }
        //}
          /*      
            for(uint32_t i = W_CSC->JA[j]; i < W_CSC->JA[j+1]; i++) {
                printf("  i=%d/%d\n", i, W_CSC->IA[i]);
                for(uint32_t k = Y_CSR->IA[W_CSC->JA[i]]; k < W_CSR->IA[Y_CSR->JA[j]+1]; k++) {
                    printf("    k=%d/",k, );
                }
                
            }
        }
   */
        
        
        
        //printf("ncols = %lu %d %d %d\n", ncols, Y_CSR->nrows, W_CSC->ncols, triples.size());
        exit(0);
        /*
        ColSort<double> f_col;
        auto f_comp = [] (const Triple<double> &a, const Triple<double> &b) {return (a.row == b.row and a.col == b.col);};    
        std::sort(triples.begin(), triples.end(), f_col);
        auto last = std::unique(triples.begin(), triples.end(), f_comp);
        triples.erase(last, triples.end());
        struct CompressedSpMat<double> ZSpMat(Y_CSR->nrows, W_CSC->ncols, triples.size(), triples, Compression_Type::dual);
        triples.clear();
        triples.shrink_to_fit();
        printf("\n");
        
        
        //ZSpMat.csr->walk();
        //exit(0);
        auto &Z = ZSpMat;
        auto *Z_CSR = Z.csr;
        //Z_CSR->walk();
        uint64_t ll = 1;
        for(uint32_t i = 0; i < Y_CSR->nrows; i++) {
            //printf("i=%d\n", i);
            for(uint32_t j = Y_CSR->IA[i]; j < Y_CSR->IA[i+1]; j++) {
                //printf("  j=%d\n", Y_CSR->JA[j]);
                for(uint32_t k = W_CSR->IA[Y_CSR->JA[j]], ll = Z_CSR->IA[i]; k < W_CSR->IA[Y_CSR->JA[j]+1], ll < Z_CSR->IA[i+1]; k++, ll++) {
                    //, l = Z_CSR->JA[Z_CSR->IA[i]]
                     //l < Z_CSR->JA[Z_CSR->IA[i+1]];
                    //printf("##############   %d %d %d %d\n", i, j, k, l);
                    //Z_CSR->A[l] += 
                    //uint32_t l = Z_CSR->IA[i];
                    Z_CSR->A[ll] += Y_CSR->A[j] * W_CSR->A[k];
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
        printf("count=%lu %lu %lu\n", Z_CSR->nnz, Z_CSR->nrows, Z_CSR->ncols);
        */
        /*
        for(uint32_t i = 0; i < Z_CSR->nrows; i++) {
            printf("i=%d\n", i);
            for(uint32_t j = Z_CSR->IA[i]; j < Z_CSR->IA[i+1]; j++) {
                  printf("i=%d, j=%d, %f\n", i, Z_CSR->JA[j], Z_CSR->A[j]);
                
            }
        }
        */
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
    struct CompressedSpMat<uint32_t> featuresSpMat((nrows + 1), (Nneurons + 1), featuresTriples.size(), featuresTriples, Compression_Type::csr_only);
    featuresTriples.clear();
    featuresTriples.shrink_to_fit();
    //featuresSpMat.csc->walk();
    //printf("\n");
    //featuresSpMat.csr->walk();
    
    //for(auto i: featuresSpMat.csc->colnrows) {
    //    printf("%d\n", i);
   // }
    
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
        //for(auto t: layerTriples) printf("%d %d %f\n", t.row, t.col, t.weight);
        //printf("%d %d %d\n", (nrows + 1), (Nneurons + 1), ncols);
        // struct CompressedSpMat<double> *layerSpMat = new struct CompressedSpMat<double>((nrows + 1), (Nneurons + 1), layerTriples.size(), layerTriples, Compression_Type::dual);
        struct CompressedSpMat<double> *layerSpMat = new struct CompressedSpMat<double>((Nneurons + 1), (ncols + 1), layerTriples.size(), layerTriples, Compression_Type::csc_only, &featuresSpMat.csr->rowncols);
        layersSpMat.push_back(layerSpMat);
        layerTriples.clear();
        layerTriples.shrink_to_fit();

        for(uint32_t j = 0; j < Nneurons; j++) {
            biasTriple.row = 1;
            biasTriple.col = j+1;
            biasTriple.weight = biasValue;
            biasTriples.push_back(biasTriple);
        }
        //struct CompressedSpMat<double> *biasSpMat = new struct CompressedSpMat<double>((nrows + 1), (Nneurons + 1), biasTriples.size(), biasTriples, Compression_Type::dual);
        struct CompressedSpMat<double> *biasSpMat = new struct CompressedSpMat<double>((Nneurons + 1), (ncols + 1), biasTriples.size(), biasTriples, Compression_Type::csc_only);
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
