/*
 * Allocator.hpp: Allocate/deallocate contiguous region of memory using mmap
 * Expand/Shrink of an already allocated memory chunk using mremap
 * To keep the realloced memory valid, we always return the new virtual address
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com
 */

#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

#include <sys/mman.h>
 
#include <unistd.h>
#include <cstring> 

template<typename Data_Type>
struct Data_Block {
    public:
        Data_Block() { ptr = nullptr; nitems = 0; nbytes = 0; }
        Data_Block(Data_Type** ptr_, uint64_t nitems_, uint64_t nbytes_, bool page_aligned_ = false);
        ~Data_Block();
        void allocate();
        void clear();
        void reallocate(Data_Type** ptr_, uint64_t nitems_, uint64_t nbytes_);
        void deallocate();
        uint64_t nitems;
        uint64_t nbytes;
        Data_Type* ptr;
        uint64_t PAGE_SIZE;
        bool page_aligned;
};

template<typename Data_Type>
Data_Block<Data_Type>::Data_Block(Data_Type** ptr_, uint64_t nitems_, uint64_t nbytes_, bool page_aligned_) {
    nitems = nitems_; 
    nbytes = nbytes_; 
    ptr = nullptr; 
    page_aligned = page_aligned_;
    PAGE_SIZE = sysconf(_SC_PAGESIZE);
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
        if(page_aligned) {
            nbytes += (PAGE_SIZE - (nbytes % PAGE_SIZE));
        }

        if((ptr = (Data_Type*) mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {  
            fprintf(stderr, "Error: Cannot map memory\n");
            exit(1);
        }
        memset(ptr, 0,  nbytes); 
    }
}

template<typename Data_Type>
void Data_Block<Data_Type>::reallocate(Data_Type** ptr_, uint64_t nitems_, uint64_t nbytes_) {
    if(nbytes) {
        if(page_aligned) {
            uint64_t new_nbytes = nbytes_;
            new_nbytes += (PAGE_SIZE - (new_nbytes % PAGE_SIZE));
            uint64_t old_nbytes = nbytes;

            if(old_nbytes != new_nbytes) {
                if((ptr = (Data_Type*) mremap(ptr, old_nbytes, new_nbytes, MREMAP_MAYMOVE)) == (void*) -1) { 
                    fprintf(stderr, "Error: Cannot remap memory\n");
                    exit(1);
                }
                if(new_nbytes > old_nbytes) {
                    memset(ptr + nitems, 0, new_nbytes - old_nbytes); // If grow zeros the added memory
                }
            }

            nitems = nitems_;
            nbytes = new_nbytes;
            *ptr_ = ptr;   
        }
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

template<typename Data_Type>
void Data_Block<Data_Type>::clear() {
    memset(ptr, 0,  nbytes); 
}
#endif