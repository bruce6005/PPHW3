#pragma once
#include <cuda_runtime.h>
void striped_sw_cuda_launcher(
    const char* ref, int refLen,
    const char* query, int queryLen,
    int match, int mismatch, int gap,
    int* DP);