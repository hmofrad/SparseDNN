/*
 * SparseMat.hpp: Sparse Matrix formats
 * Compressed Sparse Column (CSC)
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
#ifndef SPARSEMAT_HPP
#define SPARSEMAT_HPP

#include <numeric>

#include "Allocator.hpp"
#include "Triple.hpp"

template<typename Weight>
struct CSC {
    public:
        CSC() { nrows = 0, ncols = 0; nnz = 0;  nbytes = 0; idx = 0; IA = nullptr; JA = nullptr; A = nullptr; }
        CSC(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, std::vector<struct Triple<Weight>> &triples, bool page_aligned_ = false);
        ~CSC();
        void prepopulate(std::vector<struct Triple<Weight>> &triples);
        void populate(std::vector<struct Triple<Weight>> &triples);
        void postpopulate();
        void repopulate(struct CSC<Weight> *other_csc);
        void populate_spa(struct DenseVec<Weight> *SPA_Vector, uint32_t col_idx);
        void walk();
        uint64_t numnonzeros() const { return(nnz); };
        uint32_t numrows()   const { return(nrows); };
        uint32_t numcols()   const { return(ncols); };
        uint64_t size()        const { return(nbytes); };
        void clear();
        
        uint32_t nrows;
        uint32_t ncols;
        uint64_t nnz;
        uint64_t nbytes;
        uint64_t idx;
        uint32_t *IA; // Rows
        uint32_t *JA; // Cols
        Weight   *A;  // Vals
        struct Data_Block<uint32_t> *IA_blk;
        struct Data_Block<uint32_t> *JA_blk;
        struct Data_Block<Weight>  *A_blk;
};

template<typename Weight>
CSC<Weight>::CSC(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, std::vector<struct Triple<Weight>> &triples, bool page_aligned_) {
    nrows = nrows_;
    ncols = ncols_;
    nnz   = nnz_;
    IA = nullptr;
    JA = nullptr;
    A  = nullptr;
    prepopulate(triples);
    IA_blk = new Data_Block<uint32_t>(&IA, nnz, nnz * sizeof(uint32_t), page_aligned_);
    JA_blk = new Data_Block<uint32_t>(&JA, (ncols + 1), (ncols + 1) * sizeof(uint32_t), page_aligned_);
    A_blk  = new Data_Block<Weight>(&A,  nnz, nnz * sizeof(Weight), page_aligned_);
    nbytes = IA_blk->nbytes + JA_blk->nbytes + A_blk->nbytes;
    idx = 0;
    JA[0] = 0;
    populate(triples);
}

template<typename Weight>
CSC<Weight>::~CSC(){
    delete IA_blk;
    IA = nullptr;
    delete JA_blk;
    JA = nullptr;
    delete  A_blk;
    A  = nullptr;
}

template<typename Weight>
void CSC<Weight>::prepopulate(std::vector<struct Triple<Weight>> &triples) {
    uint64_t triples_size = triples.size();
    if(triples_size) {
        ColSort<Weight> f_col;
        std::sort(triples.begin(), triples.end(), f_col);
        for(uint64_t i = 1; i < triples_size; i++) {
            if(triples[i-1].col == triples[i].col) {
                if(triples[i-1].row == triples[i].row) {
                    triples[i].weight += triples[i-1].weight;
                    triples.erase(triples.begin()+i-1);
                }
            }
        }
        nnz = triples.size();
    }
}


template<typename Weight>
void CSC<Weight>::populate(std::vector<struct Triple<Weight>> &triples) {
    if(ncols and nnz and triples.size()) {
        uint32_t i = 0;
        uint32_t j = 1;        
        JA[0] = 0;
        for(auto &triple : triples) {
            while((j - 1) != triple.col) {
                j++;
                JA[j] = JA[j - 1];
            }                  
            JA[j]++;
            IA[i] = triple.row;
            A[i] = triple.weight;
            i++;
        }
        while((j + 1) < ncols) {
            j++;
            JA[j] = JA[j - 1];
        }
    }
}

template<typename Weight>
void CSC<Weight>::populate_spa(struct DenseVec<Weight> *SPA_Vector, uint32_t col_idx) {
    JA[col_idx+1] += JA[col_idx];
    for(uint32_t i = 0; i < nrows; i++) {
        auto &v = SPA_Vector->A[i];
        if(v) {
            JA[col_idx+1]++;
            IA[idx] = i;
            A[idx] = v;
            idx++;
            v = 0;
        }
    }
}

template<typename Weight>
void CSC<Weight>::postpopulate() {
    nnz = idx;
    JA_blk->reallocate(&JA, nnz, (nnz * sizeof(uint32_t)));
    A_blk->reallocate(&A, nnz, (nnz * sizeof(Weight)));
    nbytes = IA_blk->nbytes + JA_blk->nbytes + A_blk->nbytes;
}

template<typename Weight>
void CSC<Weight>::repopulate(struct CSC<Weight> *other_csc){
    uint32_t o_ncols = other_csc->numcols();
    uint32_t o_nnz = other_csc->numnonzeros();
    uint32_t *o_IA = other_csc->IA;
    uint32_t *o_JA = other_csc->JA;
    Weight   *o_A  = other_csc->A;
    if(ncols != o_ncols) {
        fprintf(stderr, "Error: Cannot repopulate CSC\n");
        exit(1);
    }
    if(nnz < o_nnz) {
        IA_blk->reallocate(&IA, o_nnz, (o_nnz * sizeof(uint32_t)));
        A_blk->reallocate(&A, o_nnz, (o_nnz * sizeof(Weight)));
    }
    clear();
    idx = 0;
    for(uint32_t j = 0; j < o_ncols; j++) {
        JA[j+1] = JA[j];
        for(uint32_t i = o_JA[j]; i < o_JA[j + 1]; i++) {
            if(o_A[i]) {
                JA[j+1]++;
                IA[idx] = o_IA[i];
                A[idx]  = o_A[i];
                idx++;
            }
        }
    }
    postpopulate();   
}

template<typename Weight>
void CSC<Weight>::clear() {
    IA_blk->clear();
    JA_blk->clear();
    A_blk->clear();
}    

template<typename Weight>
void CSC<Weight>::walk() {
    for(uint32_t j = 0; j < ncols; j++) {
        printf("j=%d\n", j);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            IA[i];
            A[i];
            std::cout << "i=" << IA[i] << ",j=" << j <<  ",value=" << A[i] << std::endl;
        }
    }
}

enum Compression_Type{ 
    csc_fmt,
    dcsc_fmt,
    tcsc
};

template<typename Weight>
struct CompressedSpMat {
    public: 
        CompressedSpMat() {csc = nullptr;};
        CompressedSpMat(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, std::vector<struct Triple<Weight>> &triples, Compression_Type type_, std::vector<uint32_t> *rowncols_ = nullptr);
        ~CompressedSpMat();
        enum Compression_Type type;
        struct CSC<Weight> *csc;
        //struct DCSC<Weight> *dcsc;
        uint64_t nbytes;
};



template<typename Weight>
CompressedSpMat<Weight>::CompressedSpMat(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, std::vector<struct Triple<Weight>> &triples, Compression_Type type_, std::vector<uint32_t> *rowncols_) {
    type = type_;
    if(type == csc_fmt) {
        csc = new CSC<Weight>(nrows_, ncols_, nnz_, triples, true);
    }
    else if(type == dcsc_fmt) {
        printf("DCSC\n");
        exit(0);
    }
    else {
        fprintf(stderr, "Error: Cannot find requested compression %d\n", type);
        exit(1);
    }
}


template<typename Weight>
CompressedSpMat<Weight>::~CompressedSpMat() {
    if(type == csc_fmt) {
        delete csc;
    }
    /*
    else if(type == dcsc) {
        delete dcsc;
    }
    */
    else {
        fprintf(stderr, "Error: Cannot find requested compression %d\n", type);
        exit(1);
    }
}

#endif
