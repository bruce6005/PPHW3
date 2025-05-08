// *** CUDA 核心代码文件: striped_sw_kernel.cu ***
#include "striped_sw.h"
#include <algorithm>  // 可选：在主机端使用 std::max 等算法

// CUDA 内核函数：执行条纹 Smith-Waterman 动态规划比对
__global__
void striped_sw_kernel(const char* d_seq1, const char* d_seq2,
                       int len1, int len2,
                       int* d_H, int* d_E,
                       int* d_bestScore, int* d_bestPos_i, int* d_bestPos_j) {
    // 每个线程处理 DP 矩阵的一行 (对应序列1的一个比对位置)
    int i = threadIdx.x;
    
    // 将 d_H 和 d_E 分为两个部分：前半部分表示上一列 (old)，后半部分表示当前列 (new)
    int *H_old = d_H;
    int *H_new = d_H + (len1 + 1);
    int *E_old = d_E;
    int *E_new = d_E + (len1 + 1);
    
    // 初始化 DP 矩阵第0列：H(i,0) = 0, E(i,0) = 0  (包括边界行i=0)
    if (i <= len1) {
        H_old[i] = 0;
        E_old[i] = 0;
    }
    
    // 使用共享内存保存当前比对的最佳得分及位置 (单个线程块内共享)
    __shared__ int bestScore;
    __shared__ int best_i;
    __shared__ int best_j;
    if (i == 0) {
        bestScore = 0;
        best_i = 0;
        best_j = 0;
    }
    __syncthreads();
    
    // 遍历参考序列的每个字符 (DP 矩阵每一列 j)
    for (int j = 1; j <= len2; ++j) {
        // 确保上一列计算完成，准备计算第 j 列
        __syncthreads();
        // 设置当前列顶部边界：H(0,j) = 0, E(0,j) = 0
        if (i == 0) {
            H_new[0] = 0;
            E_new[0] = 0;
        }
        __syncthreads();
        
        if (i > 0 && i <= len1) {
            // 取序列字符：a 为序列1第 i 个字符, b 为序列2第 j 个字符
            char a = d_seq1[i-1];
            char b = d_seq2[j-1];
            // 计算匹配/不匹配得分
            int score = (a == b ? MATCH_SCORE : MISMATCH_SCORE);
            // 计算对角线来源 (匹配/错配) 的临时得分：diag = H(i-1, j-1) + score
            int diag_score = H_old[i-1] + score;
            // 计算水平方向 gap 的得分：来自左侧单元的 H 或延续gap的 E
            int e_score = max(H_old[i] + GAP_OPEN, E_old[i] + GAP_EXTEND);
            // 计算当前单元的初始 H 值（不考虑垂直gap）：取对角或水平的较大者，与0比较
            int h_val = diag_score;
            if (e_score > h_val) {
                h_val = e_score;
            }
            if (h_val < 0) {
                // 本地比对若得分为负则置0，表示从此处重新开始比对
                h_val = 0;
            }
            // 将计算得到的 H 和 E 存入当前列数组
            H_new[i] = h_val;
            E_new[i] = e_score;
        }
        __syncthreads();
        
        // Lazy-F 垂直gap延迟处理：由线程0串行将垂直gap延伸影响传播向下更新本列
        if (i == 0) {
            int F = 0;  // F(0,j) = 0 顶端边界初始化为0
            // 遍历序列1的各行 (i = 1..len1)，计算垂直gap影响
            for (int ii = 1; ii <= len1; ++ii) {
                // 计算垂直方向得分：
                //   取上方单元的 H 减去 gap开启罚分，和上一行垂直gap F 减去 gap延伸罚分，较大者为新的 f_score
                int f_score = F + GAP_EXTEND;
                int h_above = H_new[ii-1] + GAP_OPEN;
                if (h_above > f_score) {
                    f_score = h_above;
                }
                F = f_score;  // 更新当前行的垂直gap得分 (供下一行使用)
                // 若垂直gap得分优于当前 H，则更新当前单元的 H 值
                if (F > H_new[ii]) {
                    H_new[ii] = F;
                }
                // 更新全局最佳得分和位置（如果当前单元成为新的最大值）
                if (H_new[ii] > bestScore) {
                    bestScore = H_new[ii];
                    best_i = ii;
                    best_j = j;
                }
            }
        }
        __syncthreads();
        
        // 将当前列结果复制到旧列数组，为下一列计算做准备
        if (i <= len1) {
            H_old[i] = H_new[i];
            E_old[i] = E_new[i];
        }
        // 继续循环计算下一列 j+1
    }
    
    // 全部列计算结束后，由线程0将最佳得分和对应位置写入输出变量
    if (i == 0) {
        *d_bestScore = bestScore;
        *d_bestPos_i = best_i;
        *d_bestPos_j = best_j;
    }
}

// 主机端函数：分配内存、调用内核并获取结果
void runStripedSW(const char* seq1, int len1,
                  const char* seq2, int len2,
                  int &outBestScore, int &outEndPosSeq1, int &outEndPosSeq2) {
    // 1. 分配GPU内存并将输入序列数据拷贝到设备
    char *d_seq1 = nullptr, *d_seq2 = nullptr;
    cudaMalloc((void**)&d_seq1, len1 * sizeof(char));
    cudaMalloc((void**)&d_seq2, len2 * sizeof(char));
    cudaMemcpy(d_seq1, seq1, len1 * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq2, seq2, len2 * sizeof(char), cudaMemcpyHostToDevice);
    
    // 2. 分配DP矩阵所需内存：H和E各2*(len1+1)个元素（分别存储当前列和上一列）
    int *d_H = nullptr, *d_E = nullptr;
    cudaMalloc((void**)&d_H, (len1 + 1) * 2 * sizeof(int));
    cudaMalloc((void**)&d_E, (len1 + 1) * 2 * sizeof(int));
    
    // 3. 分配输出结果内存：最佳得分和结束位置 (序列1和序列2索引)
    int *d_bestScore = nullptr, *d_bestPos_i = nullptr, *d_bestPos_j = nullptr;
    cudaMalloc((void**)&d_bestScore, sizeof(int));
    cudaMalloc((void**)&d_bestPos_i, sizeof(int));
    cudaMalloc((void**)&d_bestPos_j, sizeof(int));
    
    // 4. 配置并启动内核：使用1个线程块，线程数 = len1 + 1（序列1长度加上边界0行）
    int threads = len1 + 1;
    int blocks = 1;
    striped_sw_kernel<<<blocks, threads>>>(d_seq1, d_seq2, len1, len2,
                                           d_H, d_E,
                                           d_bestScore, d_bestPos_i, d_bestPos_j);
    cudaDeviceSynchronize();  // 等待 GPU 计算完成
    
    // 5. 将结果从设备复制回主机输出变量
    cudaMemcpy(&outBestScore, d_bestScore, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&outEndPosSeq1, d_bestPos_i, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&outEndPosSeq2, d_bestPos_j, sizeof(int), cudaMemcpyDeviceToHost);
    
    // 6. 释放GPU分配的内存
    cudaFree(d_seq1);
    cudaFree(d_seq2);
    cudaFree(d_H);
    cudaFree(d_E);
    cudaFree(d_bestScore);
    cudaFree(d_bestPos_i);
    cudaFree(d_bestPos_j);
}

