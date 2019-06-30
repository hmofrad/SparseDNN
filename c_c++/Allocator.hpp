/*
 * Allocator.hpp: Allocate contiguous region of memory using mmap
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@pitt.edu
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
        void reallocate(uint64_t nitems_);
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
    //printf("PAGE_SIZE=%lu\n" , PAGE_SIZE);
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
void Data_Block<Data_Type>::reallocate(uint64_t nbytes_) {
    if(nbytes) {
        if(page_aligned) {
            
            uint64_t new_nbytes = nbytes_;
            new_nbytes += (PAGE_SIZE - (nbytes_ % PAGE_SIZE));
            uint64_t old_nbytes = nbytes;
            printf("nitems_=%lu %lu %lu %p \n", nbytes_, old_nbytes, new_nbytes, ptr);
            Data_Type* ptr1;
            if((ptr1 = (Data_Type*) mremap(ptr, old_nbytes, new_nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) { 
                fprintf(stderr, "Error: Cannot map memory %d\n", ptr1 == MAP_FAILED );
                exit(1);
            }
            //memset(ptr, 0,  nbytes); 
            //void *p0 = mremap(p2, 4096, 4096, MREMAP_MAYMOVE | MREMAP_FIXED, p1);
    //if ( p0 == MAP_FAILED ) {
      //  perror("mremap: mremap failed");
        //return EXIT_FAILURE;
    //}
            
            
            
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
#endif