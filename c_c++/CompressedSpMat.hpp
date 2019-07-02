/*
 * CompressedSpMat.hpp: Compressed Sparse Matrix formats
 * Compressed Sparse Column (CSC)
 * Compressed Sparse Row (CSR)
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@pitt.edu
 */
 
#ifndef COMPRESSEDSPMAT_HPP
#define COMPRESSEDSPMAT_HPP

#include "Allocator.hpp"
#include "triple.hpp"

/*
template<typename Weight>
struct CompressedSpMat {
    public:
        CompressedSpMat() { nrows = 0, ncols = 0; nnz = 0; IA = nullptr; JA = nullptr; A = nullptr; }
        virtual ~CompressedSpMat() { }
        virtual void populate(std::vector<struct Triple<Weight>> &triples) {}
        virtual void walk() {}
        uint64_t numnonzeros() const { printf("parent\n"); return(nnz); }
        uint64_t numrows()   const { return(nrows); }
        uint64_t numcols()   const { return(ncols); }
        uint32_t nrows;
        uint32_t ncols;
        uint64_t nnz;
        uint32_t *IA; // Rows
        uint32_t *JA; // Cols
        Weight   *A;  // Vals
        struct Data_Block<uint32_t> *IA_blk;
        struct Data_Block<uint32_t> *JA_blk;
        struct Data_Block<Weight>   *A_blk;
};




template<typename Weight>
struct CSR : public CompressedSpMat<Weight> {{
    public:
        CSR() { nrows = 0, ncols = 0; nnz = 0; IA = nullptr; JA = nullptr; A = nullptr; }
        CSR(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_);
        virtual ~CSR();
        virtual void populate(std::vector<struct Triple<Weight>> &triples);
        virtual void walk();
        uint64_t numnonzeros() const { return(nnz); };
        uint64_t numrows()   const { return(nrows); };
        uint64_t numcols()   const { return(ncols); };
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
            while((i - 1) != triple.row) {
                i++;
                IA[i] = IA[i - 1];
            }                  
            IA[i]++;
            JA[j] = triple.col;
            A[j] = triple.weight;
            j++;
        }
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

*/




template<typename Weight>
struct CSC {
    public:
        CSC() { nrows = 0, ncols = 0; nnz = 0;  nbytes = 0; IA = nullptr; JA = nullptr; A = nullptr; }
        CSC(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_);
        ~CSC();
        void populate(std::vector<struct Triple<Weight>> &triples);
        void populate(std::vector<struct Triple<Weight>> &triples, std::vector<uint32_t> *rowncols_);
        void updaterownelems(std::vector<uint32_t> *rowncols_);
        void walk();
        uint64_t numnonzeros() const { return(nnz); };
        uint64_t numrows()   const { return(nrows); };
        uint64_t numcols()   const { return(ncols); };
        uint64_t size()        const { return(nbytes); };
        uint32_t nrows;
        uint32_t ncols;
        uint64_t nnz;
        uint64_t nbytes;
        std::vector<uint32_t>  colnrows;
        std::vector<uint32_t> rownelems;
        //std::vector<uint32_t> *rowncols;
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
    nbytes = IA_blk->nbytes + JA_blk->nbytes + A_blk->nbytes;
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
        //colnrows.resize(ncols);
        
        ColSort<Weight> f_col;
        std::sort(triples.begin(), triples.end(), f_col);
        
        uint32_t i = 0;
        uint32_t j = 1;
        JA[0] = 0;
        for(auto &triple : triples) {
            while((j - 1) != triple.col) {
               // colnrows[j-1] = JA[j] - JA[j-1];
                j++;
                JA[j] = JA[j - 1];
            }                  
            JA[j]++;
            IA[i] = triple.row;
            A[i] = triple.weight;
            i++;
        }
        //if((j + 1) > ncols) {
          //  colnrows[j-1] = JA[j] - JA[j-1];
        //}
        while((j + 1) < ncols) {
          //  colnrows[j-1] = JA[j] - JA[j-1];
            j++;
            JA[j] = JA[j - 1];
        }
    }
}

template<typename Weight>
void CSC<Weight>::populate(std::vector<struct Triple<Weight>> &triples, std::vector<uint32_t> *rowncols_) {
    if(ncols and nnz) {
        
        std::vector<uint32_t> &rowncols = *rowncols_;
        rownelems.resize(rowncols.size());
        //if(rowncols) {
            //printf("hastesh\n");
            //for(int i = 0; i < rowncols.size(); i++) {
                //printf("%d ", rowncols[i]);
            //}
            
            //for(int i = 0; i < rowncols->size(); i++) {
            //    printf("%d ", rowncols->at(2));
            //}
            
            //printf("%d %d %d\n", rowncols->operator[](2), rowncols->at(2), rowncols->size());
            
            /*
            for(auto i: *(rowncols)) {
                printf("%d ", i);
            }
            printf("\n");
            */
            /*
            std::vector<uint32_t> &r = *rowncols_;
            printf("%d\n", r[3]);
            */
            
        //}
        //else
            //printf("Nistesh\n");
        //exit(0);
        //colnrows.resize(ncols);
        
        ColSort<Weight> f_col;
        std::sort(triples.begin(), triples.end(), f_col);
        
        uint32_t i = 0;
        uint32_t j = 1;
        JA[0] = 0;
        for(auto &triple : triples) {
            while((j - 1) != triple.col) {
                //colnrows[j-1] = JA[j] - JA[j-1];
                
                if(JA[j] - JA[j-1]) {
                //if(colnrows[j-1]) {
                    updaterownelems(rowncols_);
                    /*
                    for(uint32_t i = 0; i < rowncols.size(); i++) {
                        if(rowncols[i])
                            rownelems[i]++;
                    }
                    */
                }
                
                j++;
                JA[j] = JA[j - 1];
            }                  
            JA[j]++;
            IA[i] = triple.row;
            A[i] = triple.weight;
            i++;
        }
        if((j + 1) > ncols) {
            //colnrows[j-1] = JA[j] - JA[j-1];
            
            if(JA[j] - JA[j-1]) {
            //if(colnrows[j-1]) {
                updaterownelems(rowncols_);
                /*
                for(uint32_t i = 0; i < rowncols.size(); i++) {
                    if(rowncols[i])
                        rownelems[i]++;
                }
                */
            }
        }
        while((j + 1) < ncols) {
            //colnrows[j-1] = JA[j] - JA[j-1];
            j++;
            JA[j] = JA[j - 1];
        }
        
        /*
            for(auto &i: rownelems)
                printf("%d ", i);
            printf("\n");
        //    exit(0);
        */
    }
}

template<typename Weight>
void CSC<Weight>::updaterownelems(std::vector<uint32_t> *rowncols_) {
    std::vector<uint32_t> &rowncols = *rowncols_;
    for(uint32_t i = 0; i < rowncols.size(); i++) {
        if(rowncols[i]) {
            rownelems[i]++;
        }
    }
}


template<typename Weight>
void CSC<Weight>::walk() {
    for(uint32_t j = 0; j < ncols; j++) {
        printf("j=%d\n", j);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            IA[i];
            A[i];
            std::cout << "i=" << IA[i] << ",j=" << j <<  ",value=" << A[i] << std::endl;
            //printf("    i=%d, j=%d, value=%f\n", IA[i], j, A[i]);
        }
    }
}


template<typename Weight>
struct CSR {
    public:
        CSR() { nrows = 0, ncols = 0; nnz = 0; nbytes = 0; IA = nullptr; JA = nullptr; A = nullptr; }
        //CSR(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_);
        CSR(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, bool page_aligned_ = false);
        ~CSR();
        void populate(std::vector<struct Triple<Weight>> &triples);
        void populate(struct Triple<Weight> &triple);
        void populate_spa(std::vector<struct Triple<Weight>> &triples);
        void postpopulate();
        void walk();
        uint64_t numnonzeros() const { return(nnz); };
        uint32_t numrows()   const { return(nrows); };
        uint32_t numcols()   const { return(ncols); };
        uint64_t size()        const { return(nbytes); };
        uint32_t nrows;
        uint32_t ncols;
        uint64_t nnz;
        uint64_t nbytes;
        std::vector<uint32_t> rowncols;
        std::vector<uint32_t> *rownelems;
        uint64_t idx;
        //uint64_t idx_row;
        //uint64_t idx_col;
        uint32_t *IA; // Rows
        uint32_t *JA; // Cols
        Weight   *A;  // Vals
        struct Data_Block<uint32_t> *IA_blk;
        struct Data_Block<uint32_t> *JA_blk;
        struct Data_Block<Weight>  *A_blk;
};

template<typename Weight>
CSR<Weight>::CSR(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, bool page_aligned_) {
    nrows = nrows_;
    ncols = ncols_;
    nnz   = nnz_;
    IA = nullptr;
    JA = nullptr;
    A  = nullptr;
    IA_blk = new Data_Block<uint32_t>(&IA, (nrows + 1), (nrows + 1) * sizeof(uint32_t), page_aligned_);
    JA_blk = new Data_Block<uint32_t>(&JA, nnz, nnz * sizeof(uint32_t), page_aligned_);
    A_blk  = new Data_Block<Weight>(&A,  nnz, nnz * sizeof(Weight), page_aligned_);
    nbytes = IA_blk->nbytes + JA_blk->nbytes + A_blk->nbytes;
    idx = 0;
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
        rowncols.resize(nrows);
        
        RowSort<Weight> f_row;
        std::sort(triples.begin(), triples.end(), f_row);
        
        uint32_t i = 1;
        uint32_t j = 0;
        IA[0] = 0;
        for(auto &triple : triples) {
            while((i - 1) != triple.row) {
                rowncols[i-1] = IA[i] - IA[i-1];
                i++;
                IA[i] = IA[i - 1];
            }                  
            IA[i]++;
            JA[j] = triple.col;
            A[j]  = triple.weight;
            j++;
        }
        if((i + 1) > nrows) {
            rowncols[i-1] = IA[i] - IA[i-1];
        }
        while((i + 1) < nrows) {
            rowncols[i-1] = IA[i] - IA[i-1];
            i++;
            IA[i] = IA[i - 1];
        }
    }
}

template<typename Weight>
void CSR<Weight>::populate(struct Triple<Weight> &triple) {
    IA[triple.row+1]++;
    //printf("<%d %d %f %lu %d>\n", triple.row, triple.col, triple.weight, idx, IA[triple.row+1]);
    JA[idx] = triple.col;
    A[idx] = triple.weight;
    idx++;
}

template<typename Weight>
void CSR<Weight>::populate_spa(std::vector<struct Triple<Weight>> &triples) {
    for(auto &triple: triples) {
        IA[triple.row+1]++;
        //printf("<%d %d %f %lu %d>\n", triple.row, triple.col, triple.weight, idx, IA[triple.row+1]);
        JA[idx] = triple.col;
        A[idx] = triple.weight;
        idx++;
    }
}

template<typename Weight>
void CSR<Weight>::postpopulate() {
    uint32_t i = 1;
    IA[0] = 0;
    while(i < nrows + 1) {
        //if(IA[i] == 0)
          //  IA[i] = IA[i-1];
        //else
            IA[i] += IA[i-1];
        i++;
    }
    nnz = idx;
    JA_blk->reallocate(nnz, (nnz * sizeof(uint32_t)));
    A_blk->reallocate(nnz, (nnz * sizeof(Weight)));
    nbytes = IA_blk->nbytes + JA_blk->nbytes + A_blk->nbytes;
    //
    
}
    

template<typename Weight>
void CSR<Weight>::walk() {
    for(uint32_t i = 0; i < nrows; i++) {
        //printf("i=%d\n", i);
        for(uint32_t j = IA[i]; j < IA[i + 1]; j++) {
            JA[j];
            A[j];
            //printf("    i=%d, j=%d, value=%f\n", i, JA[j], A[j]);
            //std::cout << "i=" << i << ",j=" << JA[j] <<  ",value=" << A[j] << std::endl;
            //if(A[j] == 0)
                std::cout << "i=" << i << ",j=" << JA[j] <<  ",value=" << A[j] << std::endl;
        }
    }
}


enum Compression_Type{ 
    csr_only,
    csc_only,
    dual
};

template<typename Weight>
struct CompressedSpMat {
    public: 
        CompressedSpMat() {csc = nullptr; csr = nullptr;};
        CompressedSpMat(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, std::vector<struct Triple<Weight>> &triples, Compression_Type type_, std::vector<uint32_t> *rowncols_ = nullptr);
        ~CompressedSpMat();
        enum Compression_Type type;
        struct CSC<Weight> *csc;
        struct CSR<Weight> *csr;
        uint64_t nbytes;
};

template<typename Weight>
CompressedSpMat<Weight>::CompressedSpMat(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, std::vector<struct Triple<Weight>> &triples, Compression_Type type_, std::vector<uint32_t> *rowncols_) {
    type = type_;
    if(type == csc_only) {
        csc = new CSC<Weight>(nrows_, ncols_, nnz_);
        if(rowncols_)
            csc->populate(triples, rowncols_);
        else
            csc->populate(triples);
        nbytes = csc->nbytes;
    }
    else if(type == csr_only) {
        if(rowncols_) {
            uint64_t maxnnz = std::accumulate(rowncols_->begin(), rowncols_->end(), 0);
            //printf("%lu %lu %d\n", nnz_,  rowncols_->size(), std::accumulate(rowncols_->begin(), rowncols_->end(), 0));
            //exit(0);
            csr = new CSR<Weight>(nrows_, ncols_, maxnnz, true);
            
        }
        else {
            csr = new CSR<Weight>(nrows_, ncols_, nnz_);
            csr->populate(triples);
        }
        nbytes = csr->nbytes;
    }
    else if(type == dual) {
        csc = new CSC<Weight>(nrows_, ncols_, nnz_);
        if(rowncols_)
            csc->populate(triples, rowncols_);
        else
            csc->populate(triples);
        csr = new CSR<Weight>(nrows_, ncols_, nnz_);
        csr->populate(triples);
        nbytes = csc->nbytes + csr->nbytes;
    }
    else {
        fprintf(stderr, "Error: Cannot find requested compression %d\n", type);
        exit(1);
    }
}

template<typename Weight>
CompressedSpMat<Weight>::~CompressedSpMat() {
    if(type == csc_only) {
        delete csc;
    }
    else if(type == csr_only) {
        delete csr;
    }
    else if(type == dual) {
        delete csc;
        delete csr;
    }
    else {
        fprintf(stderr, "Error: Cannot find requested compression %d\n", type);
        exit(1);
    }
}

#endif