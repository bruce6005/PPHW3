#include <cuda_runtime.h>
#include "HW2_cuda.cuh"
#include <stdio.h>
__global__
void striped_sw_kernel(
    const char* ref, int refLen,
    const char* query, int queryLen,
    int match, int mismatch, int gap,
    int* DP_flat)
{
    

    int qpos = threadIdx.x + blockIdx.x * blockDim.x;
    if (qpos > queryLen) return;

    int H = 0;
    int E = 0;
    int F = 0;


    extern __shared__ int prev_Hs[];  


    if (qpos <= queryLen) prev_Hs[qpos] = 0;
    
    __syncthreads();


    for (int r = 1; r <= refLen; ++r) {

        int diag = (qpos == 0) ? 0 : prev_Hs[qpos - 1];
        int left = (qpos == 0) ? 0 : prev_Hs[qpos];  

        int match_score = 0;
        if (qpos > 0 && qpos <= queryLen) {
            match_score = (ref[r - 1] == query[qpos - 1]) ? match : mismatch;
        }

        int h = 0;
        if (qpos > 0) {
            // printf("%d ",match_score);
            // printf("\n");
            h = max(0, diag + match_score);
            h = max(h, E + gap);
            h = max(h, F + gap);
            // printf("%d ",h);
            // printf("\n");
            // printf("\n");

            while (h < (E + gap)) {
                E += gap;
                h = max(h, E);
            }
        }
        if (qpos <= queryLen)
            DP_flat[r * (queryLen + 1) + qpos] = h;

        __syncthreads(); 
        prev_Hs[qpos] = h;
        E = h;
        F = h;

        __syncthreads();  
    }
}

void striped_sw_cuda_launcher(
    const char* ref, int refLen,
    const char* query, int queryLen,
    int match, int mismatch, int gap,
    int* DP_flat)
{
    char* d_ref;
    char* d_query;
    int* d_DP;

    cudaMalloc(&d_ref, refLen);
    cudaMalloc(&d_query, queryLen);
    cudaMalloc(&d_DP, (refLen + 1) * (queryLen + 1) * sizeof(int));

    cudaMemcpy(d_ref, ref, refLen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, queryLen, cudaMemcpyHostToDevice);
    cudaMemset(d_DP, 0, (refLen + 1) * (queryLen + 1) * sizeof(int));

    size_t shared_mem_size = (queryLen + 1) * sizeof(int);

    int threadsPerBlock = 256;
    int totalThreads = queryLen + 1;
    int numBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    striped_sw_kernel<<<numBlocks, threadsPerBlock, shared_mem_size>>>(
    d_ref, refLen, d_query, queryLen,
    match, mismatch, gap, d_DP);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();    
    cudaMemcpy(DP_flat, d_DP, (refLen + 1) * (queryLen + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_ref);
    cudaFree(d_query);
    cudaFree(d_DP);
}
