/*
 * SparseMat.hpp: Sparse Matrix formats
 * Compressed Sparse Column (CSC)
 * Doubly Compressed Sparse Column (DCSC)
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
#ifndef SPARSEMAT_HPP
#define SPARSEMAT_HPP

#include <numeric>

#include "Allocator.hpp"
#include "Triple.hpp"
#include "Env.hpp"

template<typename Weight>
struct DCSC {
    public:
        DCSC() { nrows = 0, ncols = 0; nnz = 0;  nnzcols = 0; nbytes = 0; idx = 0; IA = nullptr; JA = nullptr; A = nullptr; }
        DCSC(uint32_t nrows_, uint32_t ncols_, uint32_t nnzcols_, uint64_t nnz_, bool page_aligned_ = false);
        DCSC(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, std::vector<struct Triple<Weight>> &triples, bool page_aligned_ = false);
        ~DCSC();
        inline void prepopulate(std::vector<struct Triple<Weight>> &triples);
        inline void populate(std::vector<struct Triple<Weight>> &triples);
        inline void postpopulate();
        inline void postpopulate_t();
        inline void repopulate(struct DCSC<Weight> *other_dcsc);
        inline void spapopulate(struct DenseVec<Weight> *x_vector, struct DenseVec<Weight> *spa_vector, 
                                uint32_t col_idx, uint32_t nnzcol_idx);
        inline void spapopulate_t(struct DenseVec<Weight> *x_vector, struct DenseVec<Weight> *spa_vector, 
                                uint32_t col_idx, uint32_t nnzcol_idx, int tid);                                
        inline void walk();
        inline uint64_t numnonzeros() const { return(nnz); };
        inline uint32_t numrows()   const { return(nrows); };
        inline uint32_t numcols()   const { return(ncols); };
        inline uint32_t numnonzerocols()   const { return(nnzcols); };
        inline uint64_t size()        const { return(nbytes); };
        inline void clear();
        
        uint32_t nrows;
        uint32_t ncols;
        uint64_t nnz;
        uint32_t nnzcols;
        uint64_t nbytes;
        uint64_t idx;
        uint32_t *IA; // Rows
        uint32_t *JA; // Cols
        uint32_t *JC; // Cols Ids
        uint32_t *JB; // Col bitvector
        uint32_t *JI; // Col index JC 2 J
        Weight   *A;  // Vals
        struct Data_Block<uint32_t> *IA_blk;
        struct Data_Block<uint32_t> *JA_blk;
        struct Data_Block<uint32_t> *JC_blk;
        struct Data_Block<uint32_t> *JB_blk; 
        struct Data_Block<uint32_t> *JI_blk;
        struct Data_Block<Weight>   *A_blk;
};

template<typename Weight>
DCSC<Weight>::DCSC(uint32_t nrows_, uint32_t ncols_, uint32_t nnzcols_, uint64_t nnz_, bool page_aligned_) {
    nrows = nrows_;
    ncols = ncols_;
    nnzcols = nnzcols_;
    nnz   = nnz_;
    IA = nullptr;
    JA = nullptr;
    A  = nullptr;
    IA_blk = new Data_Block<uint32_t>(&IA, nnz, nnz * sizeof(uint32_t), page_aligned_);
    JA_blk = new Data_Block<uint32_t>(&JA, (nnzcols + 1), (nnzcols + 1) * sizeof(uint32_t), page_aligned_);
    JC_blk = new Data_Block<uint32_t>(&JC, nnzcols, nnzcols * sizeof(uint32_t), page_aligned_);
    JB_blk = new Data_Block<uint32_t>(&JB, ncols, ncols * sizeof(uint32_t), page_aligned_);
    JI_blk = new Data_Block<uint32_t>(&JI, ncols, ncols * sizeof(uint32_t), page_aligned_);
    A_blk  = new Data_Block<Weight>(&A,  nnz, nnz * sizeof(Weight), page_aligned_);
    nbytes = IA_blk->nbytes + JA_blk->nbytes + JC_blk->nbytes + JB_blk->nbytes + JI_blk->nbytes + A_blk->nbytes;
    idx = 0;
    JA[0] = 0;
}

template<typename Weight>
DCSC<Weight>::DCSC(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, std::vector<struct Triple<Weight>> &triples, bool page_aligned_) {
    nrows = nrows_;
    ncols = ncols_;
    nnz   = nnz_;
    nnzcols = 0;
    IA = nullptr;
    JA = nullptr;
    A  = nullptr;
    prepopulate(triples);
    IA_blk = new Data_Block<uint32_t>(&IA, nnz, nnz * sizeof(uint32_t), page_aligned_);
    JA_blk = new Data_Block<uint32_t>(&JA, (nnzcols + 1), (nnzcols + 1) * sizeof(uint32_t), page_aligned_);
    JC_blk = new Data_Block<uint32_t>(&JC, nnzcols, nnzcols * sizeof(uint32_t), page_aligned_);
    JB_blk = new Data_Block<uint32_t>(&JB, ncols, ncols * sizeof(uint32_t), page_aligned_);
    JI_blk = new Data_Block<uint32_t>(&JI, ncols, ncols * sizeof(uint32_t), page_aligned_);
    A_blk  = new Data_Block<Weight>(&A,  nnz, nnz * sizeof(Weight), page_aligned_);
    nbytes = IA_blk->nbytes + JA_blk->nbytes + JC_blk->nbytes + JB_blk->nbytes + JI_blk->nbytes + A_blk->nbytes;
    idx = 0;
    JA[0] = 0;
    populate(triples);
}

template<typename Weight>
DCSC<Weight>::~DCSC(){
    delete IA_blk;
    IA = nullptr;
    delete JA_blk;
    JA = nullptr;
    delete JC_blk;
    JC = nullptr;
    delete JB_blk;
    JB = nullptr;
    delete JI_blk;
    JI = nullptr;
    delete  A_blk;
    A  = nullptr;
}

template<typename Weight>
inline void DCSC<Weight>::prepopulate(std::vector<struct Triple<Weight>> &triples) {
    uint64_t triples_size = triples.size();
    if(triples_size) {
        ColSort<Weight> f_col;
        std::sort(triples.begin(), triples.end(), f_col);
        for(uint64_t i = 1; i < triples_size; i++) {
            if(triples[i-1].col == triples[i].col) {
                if(triples[i-1].row == triples[i].row) {
                    triples[i].weight += triples[i-1].weight;
                    triples.erase(triples.begin()+i-1);
                    triples_size--;
                }
            }
        }
        nnz = triples.size();
        triples_size = triples.size();
        uint32_t col_idx = triples[0].col;
        nnzcols++;
        for(uint64_t i = 1; i < triples_size; i++) {
            if(col_idx != triples[i].col) {
                col_idx = triples[i].col;
                nnzcols++;
            }
        }
    }
}

template<typename Weight>
inline void DCSC<Weight>::populate(std::vector<struct Triple<Weight>> &triples) {
    int kk = 0;
    if(ncols and nnz and triples.size()) {
        uint32_t i = 0;
        uint32_t j = 1;    
        
        JA[0] = 0;
        JC[0] = triples[0].col;
        for(auto &triple: triples) { 
            if(JC[j - 1] != triple.col) {
                j++;
                JA[j] = JA[j - 1];
                JC[j-1] = triple.col;
            }                  
            JA[j]++;
            IA[i] = triple.row;
            A[i] = triple.weight;
            i++;
        }
        for(j = 0; j < nnzcols; j++) {
            JB[JC[j]] = 1;
            JI[JC[j]] = j;
        }   
    }
}

template<typename Weight>
inline void DCSC<Weight>::spapopulate(struct DenseVec<Weight> *x_vector, struct DenseVec<Weight> *spa_vector,
                                      uint32_t col_idx, uint32_t nnzcol_idx) {
    Weight YMIN = 0;
    Weight YMAX = 32;
    Weight *x_A = x_vector->A;
    Weight *spa_A = spa_vector->A;
    Weight value = 0;
    
    JA[col_idx+1] += JA[col_idx];
    JC[col_idx] = nnzcol_idx;
    
    for(uint32_t i = 0; i < nrows; i++) {
        if(spa_A[i]) {
            JA[col_idx+1]++;
            IA[idx] = i;
            spa_A[i] += x_A[nnzcol_idx];
            if(spa_A[i] < YMIN) {
                A[idx] = YMIN;
            }
            else if(spa_A[i] > YMAX) {
                A[idx] = YMAX;
            }
            else {
                A[idx] = spa_A[i];
            }
            idx++;
            spa_A[i] = 0;
        }
    }
}

template<typename Weight>
inline void DCSC<Weight>::postpopulate() {
    JA_blk->reallocate(&JA, (nnzcols + 1), ((nnzcols + 1) * sizeof(uint32_t)));
    JC_blk->reallocate(&JC, nnzcols, (nnzcols * sizeof(uint32_t)));
    nnz = idx;
    IA_blk->reallocate(&IA, nnz, (nnz * sizeof(uint32_t)));
    A_blk->reallocate(&A, nnz, (nnz * sizeof(Weight)));
    nbytes = JA_blk->nbytes + JC_blk->nbytes + JB_blk->nbytes + JI_blk->nbytes + IA_blk->nbytes + A_blk->nbytes;
}

template<typename Weight>
inline void DCSC<Weight>::spapopulate_t(struct DenseVec<Weight> *x_vector, struct DenseVec<Weight> *spa_vector,
                                      uint32_t col_idx, uint32_t nnzcol_idx, int tid) {
    Weight YMIN = 0;
    Weight YMAX = 32;
    Weight *x_A = x_vector->A;
    Weight *spa_A = spa_vector->A;
    Weight value = 0;
    auto &idx = Env::offset_nnz[tid];
    JC[col_idx] = nnzcol_idx;
    
    for(uint32_t i = 0; i < nrows; i++) {
        if(spa_A[i]) {
            JA[col_idx+1]++;
            IA[idx] = i;
            spa_A[i] += x_A[nnzcol_idx];
            if(spa_A[i] < YMIN) {
                A[idx] = YMIN;
            }
            else if(spa_A[i] > YMAX) {
                A[idx] = YMAX;
            }
            else {
                A[idx] = spa_A[i];
            }
            idx++;
            spa_A[i] = 0;
        }
    }
}

template<typename Weight>
inline void DCSC<Weight>::postpopulate_t() {
    JA[0] = 0;
    for(uint32_t j = 1; j < nnzcols + 1; j++) {
        JA[j] += JA[j-1];
    }
    /*
    nnz = Env::offset[Env::nthreads-1];
    JA_blk->reallocate(&JA, (nnzcols + 1), ((nnzcols + 1) * sizeof(uint32_t)));
    JC_blk->reallocate(&JC, nnzcols, (nnzcols * sizeof(uint32_t)));
    IA_blk->reallocate(&IA, nnz, (nnz * sizeof(uint32_t)));
    A_blk->reallocate(&A, nnz, (nnz * sizeof(Weight)));
    nbytes = JA_blk->nbytes + JC_blk->nbytes + JB_blk->nbytes + JI_blk->nbytes + IA_blk->nbytes + A_blk->nbytes;
    */
}


template<typename Weight>
inline void DCSC<Weight>::repopulate(struct DCSC<Weight> *other_dcsc){
    uint32_t o_ncols = other_dcsc->numcols();
    uint32_t o_nnzcols = other_dcsc->numnonzerocols();
    uint32_t o_nnz = other_dcsc->numnonzeros();
    uint32_t *o_IA = other_dcsc->IA;
    uint32_t *o_JA = other_dcsc->JA;
    uint32_t *o_JC = other_dcsc->JC;
    uint32_t *o_JB = other_dcsc->JB;
    uint32_t *o_JI = other_dcsc->JI;
    Weight   *o_A  = other_dcsc->A;
    if(ncols != o_ncols) {
        fprintf(stderr, "Error: Cannot repopulate DCSC\n");
        exit(1);
    }
    
    if(nnzcols < o_nnzcols) {
        JA_blk->reallocate(&JA, (o_nnzcols + 1), ((o_nnzcols + 1) * sizeof(uint32_t)));
        JC_blk->reallocate(&JC, o_nnzcols, (o_nnzcols * sizeof(uint32_t)));
    }
    nnzcols = o_nnzcols;
    
    if(nnz < o_nnz) {
        IA_blk->reallocate(&IA, o_nnz, (o_nnz * sizeof(uint32_t)));
        A_blk->reallocate(&A, o_nnz, (o_nnz * sizeof(Weight)));
    }
    clear();
    
    idx = 0;
    uint32_t k = 0;
    bool present = false;
    for(uint32_t j = 0; j < o_nnzcols; j++) {
        present = false;
        JA[k+1] = JA[k];
        JC[k] = o_JC[j];
        JB[JC[k]] = 1;
        JI[JC[k]] = k;
        for(uint32_t i = o_JA[j]; i < o_JA[j + 1]; i++) {
            if(o_A[i]) {
                JA[k+1]++;
                IA[idx] = o_IA[i];
                A[idx]  = o_A[i];
                idx++;
                present = true;
            }
        }
        
        if(present) {
            k++;
        }
        else {
            JA[k+1] = 0;
            JB[JC[k]] = 0;
            JI[JC[k]] = 0;
            JC[k] = 0;
            nnzcols--;
        }
    }
    postpopulate(); 
}


template<typename Weight>
inline void DCSC<Weight>::clear() {
    IA_blk->clear();
    JA_blk->clear();
    JC_blk->clear();
    JB_blk->clear();
    JI_blk->clear();
    A_blk->clear();
}  

template<typename Weight>
inline void DCSC<Weight>::walk() {
    double sum = 0;
    uint64_t k = 0;
    /*
    for(uint32_t j = 0; j < ncols; j++) {
        if(JB[j]) {
            printf("%d %d\n", j, JA[JI[j] + 1] - JA[JI[j]]);
        }
        else {
            printf("%d %d\n", j, 0);
        }
    }
    */
    for(uint32_t j = 0; j < nnzcols; j++) {
      //  printf("j=%d/%d,  sz=%d\n", j, JC[j], JA[j + 1] - JA[j]);
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            IA[i];
            A[i];
            sum += A[i];
            k++;
            //std::cout << "   i=" << IA[i] << ",j=" << JC[j] <<  ",value=" << A[i] << std::endl;
        }
    }
    printf("Checksum=%f, Count=%lu\n", sum, k);
}


template<typename Weight>
struct CSC {
    public:
        CSC() { nrows = 0, ncols = 0; nnz = 0;  nbytes = 0; idx = 0; JA = nullptr; JO = nullptr; IA = nullptr; A = nullptr; }
        CSC(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, bool page_aligned_ = true);
        CSC(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, std::vector<struct Triple<Weight>> &triples, bool page_aligned_ = true);
        ~CSC();
        inline void initialize(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_);
        inline void reinitialize(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_);
        inline void prepopulate(std::vector<struct Triple<Weight>> &triples);
        inline void populate(std::vector<struct Triple<Weight>> &triples);
        inline void postpopulate();
        inline void postpopulate_t();
        inline void postpopulate_t(int tid);
        inline void repopulate(struct CSC<Weight> *other_csc);
        inline void repopulate_t();
        inline void spapopulate(struct DenseVec<Weight> *x_vector, struct DenseVec<Weight> *spa_vector, uint32_t col_idx);
        inline void spapopulate(struct DenseVec<Weight> *spa_vector, uint32_t col_idx);
        inline void spapopulate_t(struct DenseVec<Weight> *x_vector, struct DenseVec<Weight> *spa_vector, uint32_t col_idx, int tid);
        inline void walk();
        inline uint64_t numnonzeros() const { return(nnz); };
        inline uint32_t numrows()   const { return(nrows); };
        inline uint32_t numcols()   const { return(ncols); };
        inline uint64_t size()        const { return(nbytes); };
        inline void clear();
        
        uint32_t nrows;
        uint32_t ncols;
        uint64_t nnz;
        uint64_t nnzmax;
        uint64_t nbytes;
        uint64_t idx;
        uint32_t *JA; // Cols
        uint32_t *JO; // Offsets
        uint32_t *IA; // Rows
        Weight   *A;  // Vals
        struct Data_Block<uint32_t> *IA_blk;
        struct Data_Block<uint32_t> *JA_blk;
        struct Data_Block<uint32_t> *JO_blk;
        struct Data_Block<Weight>  *A_blk;
        bool page_aligned;
};

template<typename Weight>
CSC<Weight>::CSC(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, bool page_aligned_) {
    nrows = nrows_;
    ncols = ncols_;
    nnz   = nnz_;
    nnzmax   = nnz_;
    page_aligned = page_aligned_;
    JA = nullptr;
    JO = nullptr;
    IA = nullptr;
    A  = nullptr;
    if(nrows and ncols and nnz) {
        JA_blk = new Data_Block<uint32_t>(&JA, (ncols + 1), (ncols + 1) * sizeof(uint32_t), page_aligned);
        JO_blk = new Data_Block<uint32_t>(&JO, ncols, ncols * sizeof(uint32_t), page_aligned);
        IA_blk = new Data_Block<uint32_t>(&IA, nnz, nnz * sizeof(uint32_t), page_aligned);
        A_blk  = new Data_Block<Weight>(&A,  nnz, nnz * sizeof(Weight), page_aligned);
        nbytes = IA_blk->nbytes + JA_blk->nbytes + A_blk->nbytes;
        JA[0] = 0;
    }
    idx = 0;
}

template<typename Weight>
CSC<Weight>::CSC(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, std::vector<struct Triple<Weight>> &triples, bool page_aligned_) {
    nrows = nrows_;
    ncols = ncols_;
    nnz   = nnz_;
    nnzmax   = nnz_;
    nbytes = 0;
    page_aligned = page_aligned_;
    JA = nullptr;
    JO = nullptr;
    IA = nullptr;
    A  = nullptr;
    prepopulate(triples);
    if(nrows and ncols and nnz) {
        JA_blk = new Data_Block<uint32_t>(&JA, (ncols + 1), (ncols + 1) * sizeof(uint32_t), page_aligned);
        JO_blk = new Data_Block<uint32_t>(&JO, ncols, ncols * sizeof(uint32_t), page_aligned);
        IA_blk = new Data_Block<uint32_t>(&IA, nnz, nnz * sizeof(uint32_t), page_aligned);
        A_blk  = new Data_Block<Weight>(&A,  nnz, nnz * sizeof(Weight), page_aligned);
        nbytes = IA_blk->nbytes + JA_blk->nbytes + JO_blk->nbytes + A_blk->nbytes;
        JA[0] = 0;
    }
    idx = 0;
    populate(triples);
}

template<typename Weight>
CSC<Weight>::~CSC(){
    //printf("222222\n");
    //printf("JA %p\n", JA);
    delete JA_blk;
    JA = nullptr;
    //printf("4444444444 %d\n", page_aligned);    
    //int nb = ncols * sizeof(uint32_t);
    //nb += (sysconf(_SC_PAGESIZE) - (nb % sysconf(_SC_PAGESIZE)));
    //printf("JO %p %d %d\n", JO, ncols * sizeof(uint32_t), nb);
//delete JO;
  //  JO = nullptr;
    
    
    delete IA_blk;
    IA = nullptr;
    //printf("5555555555555\n");
    //printf("A %p, %d\n", A, A_blk->nbytes);
    delete  A_blk;
    A  = nullptr;
    
    
        //printf("JO %p, %d\n", JO, JO_blk->nbytes);
    
        
        //printf("<%d %d %d>\n", JO[0], JO[1], JO[JO_blk->nitems-1]);
     //   printf("<%d %d %d>\n", JO[0], JO[1], JO[JO_blk->nbytes/4]);
    
    //printf("3333333333\n");

}

template<typename Weight>
inline void CSC<Weight>::initialize(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_) {
    
    if(nnz) {
        //
        nrows = nrows_;
        ncols = ncols_;
        if(nnz_ > nnzmax) {
            printf("Reinit %d %d\n", nnz_, nnzmax);
            nnz = nnz_;
            nnzmax = nnz_;
            JA_blk->reallocate(&JA, (ncols + 1), ((ncols + 1) * sizeof(uint32_t)));
            JO_blk->reallocate(&JO, ncols, (ncols * sizeof(uint32_t)));            
            IA_blk->reallocate(&IA, nnz, (nnz * sizeof(Weight)));
            A_blk->reallocate(&A, nnz, (nnz * sizeof(Weight)));
            nbytes = JA_blk->nbytes + JO_blk->nbytes + IA_blk->nbytes + A_blk->nbytes;
            JA[0] = 0;
        }   
        idx = 0;
        nnz = nnz_;
        clear();
        // We don not shrink
    }
    else {
        nrows = nrows_;
        ncols = ncols_;
        nnz = nnz_;
        nnzmax = nnz_;
        JA_blk = new Data_Block<uint32_t>(&JA, (ncols + 1), (ncols + 1) * sizeof(uint32_t), page_aligned);
        JO_blk = new Data_Block<uint32_t>(&JO, ncols, ncols * sizeof(uint32_t), page_aligned);
        IA_blk = new Data_Block<uint32_t>(&IA, nnz, nnz * sizeof(uint32_t), page_aligned);
        A_blk  = new Data_Block<Weight>(&A,  nnz, nnz * sizeof(Weight), page_aligned);
        nbytes = IA_blk->nbytes + JO_blk->nbytes + JA_blk->nbytes + A_blk->nbytes;
        JA[0] = 0;
        idx = 0;
    }
}

template<typename Weight>
inline void CSC<Weight>::prepopulate(std::vector<struct Triple<Weight>> &triples) {
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
inline void CSC<Weight>::populate(std::vector<struct Triple<Weight>> &triples) {
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
            JO[j] = 0;
            IA[i] = triple.row;
            A[i] = triple.weight;
            i++;
        }
        while((j + 1) <= ncols) {
            j++;
            JA[j] = JA[j - 1];
            JO[j] = 0;
        }
    }
}

template<typename Weight>
inline void CSC<Weight>::spapopulate(struct DenseVec<Weight> *x_vector, struct DenseVec<Weight> *spa_vector, uint32_t col_idx) {
    Weight YMIN = 0;
    Weight YMAX = 32;
    Weight   *x_A = x_vector->A;
    Weight   *spa_A = spa_vector->A;
    Weight value = 0;
    JA[col_idx+1] += JA[col_idx];
    for(uint32_t i = 0; i < nrows; i++) {
        if(spa_A[i]) {
            JA[col_idx+1]++;
            IA[idx] = i;
            spa_A[i] += x_A[col_idx];
            if(spa_A[i] < YMIN) {
                A[idx] = YMIN;
            }
            else if(spa_A[i] > YMAX) {
                A[idx] = YMAX;
            }
            else {
                A[idx] = spa_A[i];
            }
            idx++;
            spa_A[i] = 0;
        }
    }
}

template<typename Weight>
inline void CSC<Weight>::spapopulate(struct DenseVec<Weight> *spa_vector, uint32_t col_idx) {
    Weight YMIN = 0;
    Weight YMAX = 32;
    Weight   *spa_A = spa_vector->A;
    Weight value = 0;
    JA[col_idx+1] += JA[col_idx];
    for(uint32_t i = 0; i < nrows; i++) {
        if(spa_A[i]) {
            JA[col_idx+1]++;
            IA[idx] = i;
            A[idx] = spa_A[i];
            idx++;
            spa_A[i] = 0;
        }
    }
}

template<typename Weight>
inline void CSC<Weight>::postpopulate() {
    /*
    nnz = idx;
    JA_blk->reallocate(&JA, (ncols + 1), ((ncols + 1) * sizeof(uint32_t)));
    IA_blk->reallocate(&IA, nnz, (nnz * sizeof(Weight)));
    A_blk->reallocate(&A, nnz, (nnz * sizeof(Weight)));
    nbytes = JA_blk->nbytes + IA_blk->nbytes + A_blk->nbytes;
    */
}

template<typename Weight>
inline void CSC<Weight>::repopulate(struct CSC<Weight> *other_csc){
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
    nnz = idx;
    //postpopulate();   
}

template<typename Weight>
inline void CSC<Weight>::spapopulate_t(struct DenseVec<Weight> *x_vector, struct DenseVec<Weight> *spa_vector, uint32_t col_idx, int tid) {
    Weight YMIN = 0;
    Weight YMAX = 32;
    Weight   *x_A = x_vector->A;
    Weight   *spa_A = spa_vector->A;
    Weight value = 0;
    auto &idx = Env::offset_nnz[tid];
    //auto &off = (tid == Env::nthreads - 1) ? Env::start_nnz[tid];
    
    for(uint32_t i = 0; i < nrows; i++) {
        if(spa_A[i]) {
            spa_A[i] += x_A[col_idx];
            if(spa_A[i] < YMIN) {
                spa_A[i] = YMIN;
            }
            else if(spa_A[i] > YMAX) {
                spa_A[i] = YMAX;
            }
            if(spa_A[i]) {
                JA[col_idx+1]++;
                IA[idx] = i;
                A[idx] = spa_A[i];
                idx++;
                spa_A[i] = 0;
            }
        }
    }
    
    /*
    for(uint32_t i = 0; i < nrows; i++) {
        if(spa_A[i]) {
            JA[col_idx+1]++;
            IA[idx] = i;
            spa_A[i] += x_A[col_idx];
            if(spa_A[i] < YMIN) {
                A[idx] = YMIN;
            }
            else if(spa_A[i] > YMAX) {
                A[idx] = YMAX;
            }
            else {
                A[idx] = spa_A[i];
            }
            idx++;
            spa_A[i] = 0;
        }
    }
    */
}

template<typename Weight>
inline void CSC<Weight>::postpopulate_t(int tid) {
    uint32_t start_col = Env::start_col[tid];
    uint32_t end_col = Env::end_col[tid];
    uint32_t start_nnz = Env::start_nnz[tid];
    uint32_t end_nnz = Env::end_nnz[tid];
    uint32_t length_nnz = Env::length_nnz[tid];
    uint32_t offset_nnz = Env::offset_nnz[tid];
    /*
    if(tid == 3) {
     for(uint32_t j = start_col; j < end_col; j++) {
        printf("%d %d\n", j, JA[j]);         
    }
    }
    */
    
    //printf("1.%d %d %d\n", tid, start_col, JA[start_col]);
    
    
    //start_col++;
    //if(tid == 0) {
      //  start_col++;
    //}
    //if((tid == Env::nthreads - 1)) {
        //JA[0] = 0;
      //  end_col++;
    //}
    
    //printf("2.%d %d\n", tid, JA[start_col]);
    if(tid == 0) {
        JA[0] = 0;
    for(uint32_t j = start_col+1; j < end_col; j++) {
        
        //if(tid == 1) {
            //printf("1.%d %d %d\n", j, JA[j-1], JA[j], JA[j-1], JA[j]);
          //  JA[j] += JA[j-1];
            //printf("2.%d %d %d\n", j, JA[j-1], JA[j]);
            //exit(0);
        //}
        
        JA[j] += JA[j-1];
        JO[j] = 0;
    }
    //JO[end_col-1] = Env::end_nnz[tid];
    }
    else {
        //JO[start_col] = Env::start_nnz[tid];
        for(int32_t i = 0; i < tid; i++) {
            JA[start_col] += (Env::offset_nnz[i] - Env::start_nnz[i]);
            JO[start_col] += (Env::end_nnz[i] - Env::offset_nnz[i]);
            
        }
        printf(">>>tid=%d ja=%d jo=%d\n", tid, JA[start_col], JO[start_col]);
        for(uint32_t j = start_col+1; j < end_col; j++) {
            JA[j] += JA[j-1];
            JO[j] = JO[j-1]; //Env::end_nnz[tid-1] - Env::offset_nnz[tid-1];
        }
        
        
    }
    printf("tid=%d st=%d end=%d len=%d nnz=%d z=%d\n", tid, Env::start_nnz[tid], Env::end_nnz[tid], Env::end_nnz[tid] - Env::start_nnz[tid], Env::offset_nnz[tid], Env::end_nnz[tid] - Env::offset_nnz[tid]);
    printf("tid=%d ja=%d jo=%d\n", tid, JA[start_col], JO[start_col]);
    if((tid == Env::nthreads - 1)) {
        JA[end_col+1] += JA[end_col];
        //JA[0] = 0;
     //   end_col++;
    }
   // if(tid == 0)
     //   exit(0);
    
    if(tid == 0) {
        idx = 0;
        for(uint32_t i = 0; i < Env::nthreads; i++) {    
            idx += (Env::offset_nnz[i] - Env::start_nnz[i]);
        }
        nnz = idx;
        printf(" NNNNNZZ %d\n", nnz);    
        //exit(0);
    }
    
    

//printf("%d\n", nnz);
    /*
    else if((tid == Env::ntthreads - 1)) {
        
    }
    else {
        JA[start_col] += Env::start_nnz[tid];
        end_col++;
    }
    
    for(uint32_t j = start_col+1; j < end_col; j++) {
        JA[j] += JA[j-1];
    }
    
    */
    
    
    //exit(0);
    
    /*
    if(tid == 0) {
        JA[0] = 0;
        for(uint32_t j = start_col+1; j < end_col; j++) {
            JA[j] += JA[j-1];
        }
        //Env::indices[tid] = Env::offset[tid];
    }
    else if((tid == Env::ntthreads - 1)) {
        JA[start_col] += Env::start_nnz[tid];
        for(uint32_t j = start_col; j < end_col; j++) {
            JA[j] += JA[j-1];
        }
        
        Env::indices[tid] = Env::offset[tid];
        
    }
    else {
        
    }
    */
    
    
    /*
        JA[0] = 0;
        for(uint32_t j = start+1; j < end; j++) {
            JA[j] += JA[j-1];
        }
        */
        /*
    uint32_t start = Env::start[tid];
    uint32_t end = Env::end[tid];
    uint32_t offset = Env::offset[tid];
    uint32_t length = Env::length[tid];
    
    if(tid == 0) {
        
        Env::indices[tid] = Env::offset[tid];
    }
    else if((tid == Env::ntthreads - 1)) {
        
        Env::indices[tid] = Env::offset[tid];
        
    }
    else {
        
    }
    
    idx = Env::offset[0];
    uint64_t length = Env::length[0];
    for(uint32_t i = 1; i < Env::nthreads; i++) {
        
        idx += (Env::offset[i] - length);
        length += Env::length[i];
    }
    nnz = idx;
    */
}

template<typename Weight>
inline void CSC<Weight>::postpopulate_t() {
    JA[0] = 0;
    for(uint32_t j = 1; j < ncols+1; j++) {
        JA[j] += JA[j-1];
    }

    
    //printf("%lu %lu\n", nnzmax, Env::offset[Env::nthreads-1]);
    //exit(0);
    
    
    /*
    nnz = Env::offset[Env::nthreads-1];
    JA_blk->reallocate(&JA, (ncols + 1), ((ncols + 1) * sizeof(uint32_t)));
    IA_blk->reallocate(&IA, nnz, (nnz * sizeof(Weight)));
    A_blk->reallocate(&A, nnz, (nnz * sizeof(Weight)));
    nbytes = JA_blk->nbytes + IA_blk->nbytes + A_blk->nbytes;
    */
}

template<typename Weight>
inline void CSC<Weight>::repopulate_t() {
    /*
    idx = 0;
    JA[0] = 0;
    for(uint32_t j = 0; j < ncols; j++) {
        //JA[j+1] = JA[j];
        for(uint32_t i = JA[j]; i < JA[j + 1]; i++) {
            if(A[i]) {
                JA[j+1]++;
                IA[idx] = o_IA[i];
                A[idx]  = o_A[i];
                idx++;
            }
        }
    }
    postpopulate(); 
    */
}

template<typename Weight>
inline void CSC<Weight>::clear() {
    JA_blk->clear();
    JO_blk->clear();
    IA_blk->clear();
    A_blk->clear();
}    

template<typename Weight>
inline void CSC<Weight>::walk() {
    double sum = 0;
    uint64_t k = 0;
    uint64_t k1 = 0;
    for(uint32_t j = 0; j < ncols; j++) {
        k1 += (JA[j + 1] - JA[j]);
        //printf("j=%d jai=%d jai+1=%d joj=%d sz=%d\n", j, JA[j]+JO[j], JA[j+1]+JO[j], JO[j], JA[j + 1] - JA[j]);
//        printf("j=%d, sz=%d\n", j, JA[j + 1] - JA[j]);
        for(uint32_t i = JA[j]+JO[j]; i < JA[j + 1]+JO[j]; i++) {
            IA[i];
            A[i];
            sum += A[i];
            k++;
            //std::cout << "i=" << IA[i] << ",j=" << j <<  ",value=" << A[i] << std::endl;
        }
        
        //if(j == 300)
        //    break;
    }
    printf("Checksum=%f, Count=%lu %lu\n", sum, k, k1);
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
        CompressedSpMat(uint32_t nrows_, uint32_t ncols_, uint32_t nnzcols_, uint64_t nnz_, Compression_Type type_);
        CompressedSpMat(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, std::vector<struct Triple<Weight>> &triples, Compression_Type type_);
        ~CompressedSpMat();
        enum Compression_Type type;
        uint32_t nrows;
        uint32_t ncols;
        struct CSC<Weight> *csc;
        struct DCSC<Weight> *dcsc;
};

template<typename Weight>
CompressedSpMat<Weight>::CompressedSpMat(uint32_t nrows_, uint32_t ncols_, uint32_t nnzcols_, uint64_t nnz_, Compression_Type type_) {
    type = type_;
    if(type == csc_fmt) {
        csc = new CSC<Weight>(nrows_, ncols_, nnz_, true);
    }
    else if(type == dcsc_fmt) {
        dcsc = new DCSC<Weight>(nrows_, ncols_, nnzcols_, nnz_, true);
    }
    else {
        fprintf(stderr, "Error: Cannot find requested compression %d\n", type);
        exit(1);
    }
}

template<typename Weight>
CompressedSpMat<Weight>::CompressedSpMat(uint32_t nrows_, uint32_t ncols_, uint64_t nnz_, std::vector<struct Triple<Weight>> &triples, Compression_Type type_) {
    nrows = nrows_;
    ncols = ncols_;
    type = type_;
    if(type == csc_fmt) {
        csc = new CSC<Weight>(nrows_, ncols_, nnz_, triples, true);
    }
    else if(type == dcsc_fmt) {
        dcsc = new DCSC<Weight>(nrows_, ncols_, nnz_, triples, true);
    }
    else {
        fprintf(stderr, "Error: Cannot find requested compression %d\n", type);
        exit(1);
    }
}


template<typename Weight>
CompressedSpMat<Weight>::~CompressedSpMat() {
    if(type == csc_fmt) {
        printf("11111\n");
        delete csc;
    }
    else if(type == dcsc_fmt) {
        delete dcsc;
    }
    
    else {
        fprintf(stderr, "Error: Cannot find requested compression %d\n", type);
        exit(1);
    }
}

#endif
