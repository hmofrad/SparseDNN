/*
 * CompressedSpMat.hpp: Compressed Sparse Matrix formats
 * Compressed Sparse Column (CSC)
 * Compressed Sparse Row (CSR)
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@pitt.edu
 */
 
#ifndef CompressedSpMat_HPP
#define CompressedSpMat_HPP

#include "Allocator.hpp"
#include "triple.hpp"

template<typename Weight>
struct CSC {
    public:
        CSC() { nrows = 0, ncols = 0; nnz = 0; IA = nullptr; JA = nullptr; A = nullptr; };
        CSC(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_);
        ~CSC();
        void populate(std::vector<struct Triple<Weight>> &triples);
        void walk();
        uint64_t numnonzeros() const { return(nnz); };
        uint64_t numofrows()   const { return(nrows); };
        uint64_t numofcols()   const { return(ncols); };
        uint32_t nrows;
        uint32_t ncols;
        uint64_t nnz;
        uint32_t *IA; // Rows
        uint32_t *JA; // Cols
        Weight   *A;  // Vals
        struct Data_Block<uint32_t> *IA_blk;
        struct Data_Block<uint32_t> *JA_blk;
        struct Data_Block<Weight>  *A_blk;
        
};

template<typename Weight>
CSC<Weight>::CSC(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_) {
    nrows = nrows_;
    ncols = ncols_;
    nnz   = nnz_;
    IA = nullptr;
    JA = nullptr;
    A  = nullptr;
    IA_blk = new Data_Block<uint32_t>(&IA, nnz, nnz * sizeof(uint32_t));
    JA_blk = new Data_Block<uint32_t>(&JA, (ncols + 1), (ncols + 1) * sizeof(uint32_t));
    A_blk  = new Data_Block<Weight>(&A,  nnz, nnz * sizeof(Weight));
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
void CSC<Weight>::populate(std::vector<struct Triple<Weight>> &triples) {
    if(ncols and nnz) {
        ColSort<Weight> f_col;
        std::sort(triples.begin(), triples.end(), f_col);
        
        uint32_t i = 0;
        uint32_t j = 1;
        JA[0] = 0;
        for(auto& triple : triples) {
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
void CSC<Weight>::walk() {
    for(uint32_t j = 0; j < ncols; j++) {
        printf("j=%d\n", j);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            printf("    i=%d, j=%d, value=%f\n", IA[i], j, A[i]);
        }
        
    }
}

template<typename Weight>
struct CSR {
    public:
        CSR() { nrows = 0, ncols = 0; nnz = 0; IA = nullptr; JA = nullptr; A = nullptr; };
        CSR(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_);
        ~CSR();
        void populate(std::vector<struct Triple<Weight>> &triples);
        void walk();
        uint64_t numnonzeros() const { return(nnz); };
        uint64_t numofrows()   const { return(nrows); };
        uint64_t numofcols()   const { return(ncols); };
        uint32_t nrows;
        uint32_t ncols;
        uint64_t nnz;
        uint32_t *IA; // Rows
        uint32_t *JA; // Cols
        Weight   *A;  // Vals
        struct Data_Block<uint32_t> *IA_blk;
        struct Data_Block<uint32_t> *JA_blk;
        struct Data_Block<Weight>  *A_blk;
        
};

template<typename Weight>
CSR<Weight>::CSR(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_) {
    nrows = nrows_;
    ncols = ncols_;
    nnz   = nnz_;
    IA = nullptr;
    JA = nullptr;
    A  = nullptr;
    IA_blk = new Data_Block<uint32_t>(&IA, (nrows + 1), (nrows + 1) * sizeof(uint32_t));
    JA_blk = new Data_Block<uint32_t>(&JA, nnz, nnz * sizeof(uint32_t));
    A_blk  = new Data_Block<Weight>(&A,  nnz, nnz * sizeof(Weight));
}

template<typename Weight>
CSR<Weight>::~CSR(){
    delete IA_blk;
    IA = nullptr;
    delete JA_blk;
    JA = nullptr;
    delete  A_blk;
    A  = nullptr;
}

template<typename Weight>
void CSR<Weight>::populate(std::vector<struct Triple<Weight>> &triples) {
    if(nrows and nnz) {
        RowSort<Weight> f_row;
        std::sort(triples.begin(), triples.end(), f_row);
        
        
        uint32_t i = 1;
        uint32_t j = 0;
        IA[0] = 0;
        for(auto& triple : triples) {
            //printf("%d %d %d %d\n", triple.row, triple.col,i ,j);
            while((i - 1) != triple.row) {
                i++;
                IA[i] = IA[i - 1];
            }                  
            IA[i]++;
            JA[j] = triple.col;
            A[j] = triple.weight;
            j++;
         //   */
        }
        printf("xxxx\n");
        while((i + 1) < nrows) {
            i++;
            IA[i] = IA[i - 1];
        }
    }
    
}

template<typename Weight>
void CSR<Weight>::walk() {
    for(uint32_t i = 0; i < nrows; i++) {
        printf("i=%d\n", i);
        for(uint32_t j = IA[i]; j < IA[i + 1]; j++) {
            printf("    i=%d, j=%d, value=%f\n", i, JA[j], A[j]);
        }
    }
}


enum Compression_Type{ 
    csr_only,
    csc_only,
    dual
};

template<typename Weight>
struct Sparse {
    public: 
        Sparse() {csc = nullptr; csr = nullptr;};
        Sparse(Compression_Type type_);
        ~Sparse() {};
        enum Compression_Type type;
        struct CSC<Weight> *csc;
        struct CSC<Weight> *csr;
};

template<typename Weight>
Sparse<Weight>::Sparse(Compression_Type type_) {
    type = type_;
    if(type == csr_only) {
        printf("csr_only=%d\n", type);
    }
    else if(type == csc_only) {
        printf("csc_only=%d\n", type);
    }
    else if(type == dual) {
        printf("dual=%d\n", type);
    }
    else {
        printf("Error=%d\n", type);
    }
}

#endif