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
void Data_Block<Data_Type>::reallocate(Data_Type** ptr_, uint64_t nitems_, uint64_t nbytes_) {
    if(nbytes) {
        if(page_aligned) {
            uint64_t new_nbytes = nbytes_;
            new_nbytes += (PAGE_SIZE - (new_nbytes % PAGE_SIZE));
            uint64_t old_nbytes = nbytes;
            
            //printf("old_nbytes=%lu new_nbytes=%lu\n",  old_nbytes, new_nbytes);
            //printf("nitems_=%lu nbytes_=%lu old_nbytes=%lu new_nbytes=%lu\n",  nitems, nbytes, old_nbytes, new_nbytes);
            //if(old_nbytes < new_nbytes) { // Shrink
              //  if((ptr = (Data_Type*) mremap(ptr, old_nbytes, new_nbytes, MREMAP_MAYMOVE)) == (void*) -1) { 
                //    fprintf(stderr, "Error: Cannot remap memory\n");
                 //   exit(1);
                //}
            //}
            //else { //Expand
                
            //}
            
            
            if(old_nbytes != new_nbytes) {
                //mremap(mapping, oldsize, newsize, 0)
//                Data_Type* ptr1;
//ptr                = (Data_Type*) mremap(ptr, old_nbytes, new_nbytes, MREMAP_MAYMOVE);
  
                //if ( ptr1 == MAP_FAILED ) {
                  //  perror("mremap: mremap failed");
                    //exit(EXIT_FAILURE);
                //}
            //    printf("1.%p %p\n", ptr, ptr+nitems - 1);
                if((ptr = (Data_Type*) mremap(ptr, old_nbytes, new_nbytes, MREMAP_MAYMOVE)) == (void*) -1) { 
                    fprintf(stderr, "Error: Cannot remap memory\n");
                    exit(1);
                }
              //  printf("2.%p %p\n", ptr, ptr+nitems_ - 1);
                //memset(ptr, 0, new_nbytes);
                //printf("1.%d %d %p %lu %lu %f %p\n", nitems_, nitems, ptr, old_nbytes, new_nbytes, ptr[nitems_-1], ptr + old_nbytes);
                if(new_nbytes > old_nbytes) {
                    //uint64_t diff = new_nbytes - old_nbytes;
                    //printf("here??? %p %p %p %d\n", ptr, ptr + old_nbytes, ptr + (new_nbytes - old_nbytes), (new_nbytes - old_nbytes)/PAGE_SIZE);
                    //printf("nitems_=%lu %p %d %p %p\n", nitems_, ptr, ptr[nitems_-1], (ptr + nitems_-1), (ptr + new_nbytes - 1));
                //    printf("ptr=%p ptr+n=%p, sz=%ld\n", ptr, ptr+nitems, new_nbytes - old_nbytes);
                    memset(ptr + nitems, 0, new_nbytes - old_nbytes);
                    //memset(ptr, 0, new_nbytes);
                    //printf("done???\n");
                }
                //  printf("nitems_=%lu %lu %lu %p %d\n", nbytes_, old_nbytes, new_nbytes, ptr, ptr[2047]);
                
                  
            }
            
            //memset(ptr, 0,  nbytes); 
            //void *p0 = mremap(p2, 4096, 4096, MREMAP_MAYMOVE | MREMAP_FIXED, p1);
    //if ( p0 == MAP_FAILED ) {
      //  perror("mremap: mremap failed");
        //return EXIT_FAILURE;
    //}
        nitems = nitems_;
        nbytes = new_nbytes;
        //printf("nitems=%lu nbytes=%lu old_nbytes=%lu new_nbytes=%lu\n",  nitems, nbytes, old_nbytes, new_nbytes);
        
            
            
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