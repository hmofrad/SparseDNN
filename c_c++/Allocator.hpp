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
        Data_Block(){ ptr = nullptr; nitems = 0; nbytes = 0; };// printf("Construct \n");};
        Data_Block(Data_Type** ptr_, uint64_t nitems_, uint64_t nbytes_);
        ~Data_Block(); //{ printf("Destruct\n"); };
        //~Data_Block(Data_Type** ptr_);
        void allocate();
        void deallocate();
        uint64_t nitems;
        uint64_t nbytes;
        Data_Type* ptr;
};

template<typename Data_Type>
Data_Block<Data_Type>::Data_Block(Data_Type** ptr_, uint64_t nitems_, uint64_t nbytes_) {
    //printf("Allocate  %lu %lu %d %d %d\n",  nitems_, nbytes_, sizeof(Data_Type), sizeof(Data_Type*), sizeof(Data_Type**));
    //printf("Allocate:   <%p %p> <%d>\n", ptr_, *ptr_, sizeof(ptr));

    
    //printf("1.%p %p\n", ptr, *ptr_);
    nitems = nitems_; 
    nbytes = nbytes_; 
    ptr = nullptr; 
    //printf("Allocate %p- %lu %lu \n", *ptr, nitems, nbytes);
    allocate();
    *ptr_ = ptr;
    //printf("2.%p %p\n", ptr, *ptr_);
    //printf("Done Allocate\n");
    //printf("1.Done Allocate\n");
}

template<typename Data_Type>
Data_Block<Data_Type>::~Data_Block() {
    //printf("<%x %p   %d>\n", ptr, *ptr, **ptr);
    deallocate();
    //*ptr_ = *ptr;
    //printf("<%p   %d>\n", *ptr, **ptr);
}

template<typename Data_Type>
//void allocate(struct Data_Block<Data_Type> blk) {
void Data_Block<Data_Type>::allocate() {
    if(nbytes) {
        if((ptr = (Data_Type*) mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) == (void*) -1) {  
            fprintf(stderr, "Error mapping memory\n");
            exit(1);
        }
        memset(ptr, 0,  nbytes); 
    }
}

template<typename Data_Type>
//void deallocate(struct Data_Block<Data_Type> blk) { 
void Data_Block<Data_Type>::deallocate() {
    //printf("deallocate\n");
    if(ptr and nbytes) {
        if((munmap(ptr, nbytes)) == -1) {
            fprintf(stderr, "Error unmapping memory\n");
            exit(1);
        }
        ptr = nullptr;
    }
}

#endif