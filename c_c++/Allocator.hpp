/*
 * Allocator.hpp: Allocate contiguous region of memory using mmap
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@pitt.edu
 */

#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP
 
 
#include <sys/mman.h>
#include <cstring> 

template<typename Data_Type>
struct Data_Block {
    public:
        Data_Block() { ptr = nullptr; nitems = 0; nbytes = 0; }
        Data_Block(Data_Type** ptr_, uint64_t nitems_, uint64_t nbytes_);
        ~Data_Block();
        void allocate();
        void deallocate();
        uint64_t nitems;
        uint64_t nbytes;
        Data_Type* ptr;
};

template<typename Data_Type>
Data_Block<Data_Type>::Data_Block(Data_Type** ptr_, uint64_t nitems_, uint64_t nbytes_) {
    nitems = nitems_; 
    nbytes = nbytes_; 
    ptr = nullptr; 
    allocate();
    *ptr_ = ptr;
}

template<typename Data_Type>
Data_Block<Data_Type>::~Data_Block() {
    deallocate();
}

template<typename Data_Type>
void Data_Block<Data_Type>::allocate() {
    if(nbytes) {
        if((ptr = (Data_Type*) mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {  
            fprintf(stderr, "Error: Cannot map memory\n");
            exit(1);
        }
        memset(ptr, 0,  nbytes); 
    }
}

template<typename Data_Type>
void Data_Block<Data_Type>::deallocate() {
    if(ptr and nbytes) {
        if((munmap(ptr, nbytes)) == -1) {
            fprintf(stderr, "Error: Cannot unmap memory\n");
            exit(1);
        }
        ptr = nullptr;
    }
}
#endif