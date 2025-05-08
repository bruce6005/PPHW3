// 条纹 Smith-Waterman CUDA 实现范例
// *** 头文件: striped_sw.h ***
#ifndef STRIPED_SW_H
#define STRIPED_SW_H

#include <cuda_runtime.h>  // CUDA 运行时库头文件

// 定义配分常数 (匹配、错配和缺口罚分)
const int MATCH_SCORE   = 2;
const int MISMATCH_SCORE = -1;
const int GAP_OPEN     = -3;   // 缺口开启罚分 (负值，表示扣分)
const int GAP_EXTEND   = -1;   // 缺口延伸罚分 (负值)

// 声明用于在 GPU 上执行条纹 Smith-Waterman 比对的函数。
// 该函数执行比对并返回最佳比对得分和对应的结束位置 
// （outEndPosSeq1 和 outEndPosSeq2 分别为序列1和序列2中的索引位置）。
void runStripedSW(const char* seq1, int len1,
                  const char* seq2, int len2,
                  int &outBestScore, int &outEndPosSeq1, int &outEndPosSeq2);

#endif // STRIPED_SW_H

