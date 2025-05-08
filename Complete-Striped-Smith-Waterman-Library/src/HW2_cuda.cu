
// HW2_cuda.cu
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
    
    // 一個 block，queryLen 條 thread，每條 thread 對應 query 的一個位置
    int qpos = threadIdx.x + blockIdx.x * blockDim.x;
    if (qpos > queryLen) return;
    // 這裡的 H、E、F 跟 scalar 或 SIMD 一樣
    int H = 0;
    int E = 0;
    int F = 0;

    // 每個 thread 有一個自己的 DP column
    extern __shared__ int prev_Hs[];  // 上一 row 的 H 值

    // 初始化 prev_Hs
    if (qpos <= queryLen) prev_Hs[qpos] = 0;
    
    __syncthreads();

    // row by row
    for (int r = 1; r <= refLen; ++r) {

        int diag = (qpos == 0) ? 0 : prev_Hs[qpos - 1];
        int left = (qpos == 0) ? 0 : prev_Hs[qpos];  // prev row same column

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
            // Lazy loop (模擬 xsimd 的 lazy-F loop)
            while (h < (E + gap)) {
                E += gap;
                h = max(h, E);
            }
        }

        // 更新 DP
        if (qpos <= queryLen)
            DP_flat[r * (queryLen + 1) + qpos] = h;

        // 保存 H 給下一 row
        __syncthreads();  // 保證所有 thread 都算完
        prev_Hs[qpos] = h;

        // 更新 E、F
        E = h;
        F = h;

        __syncthreads();  // 保證下一 row 前同步
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

    // 每一個 query position 對應一個 thread
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
