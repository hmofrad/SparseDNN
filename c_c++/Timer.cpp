/*
 * Timer.cpp: Timer implementation
 * (C) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com
 */
 
#ifndef TIMER_CPP
#define TIMER_CPP

double elapsed_time;
std::chrono::high_resolution_clock::time_point start, finish;

void tic() { 
    start1 = std::chrono::high_resolution_clock::now();
}
void toc(std::string str = " ");
void toc(std::string str) { 
    finish = std::chrono::high_resolution_clock::now();
    elapsed_time = (double)(std::chrono::duration_cast< std::chrono::nanoseconds>(finish-start).count())/1e9;
    printf("elapsed_time - %s =%f\n", str.c_str(), elapsed_time);
}

#endif