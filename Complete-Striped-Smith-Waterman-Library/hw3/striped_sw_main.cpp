// *** 主程序文件: striped_sw_main.cpp ***
#include <iostream>
#include <string>
#include "striped_sw.h"

int main() {
    // 准备两条待比对的示例序列
    std::string seq1 = "ACACACTA";
    std::string seq2 = "AGCACACA";
    int bestScore = 0;
    int endPosSeq1 = 0;
    int endPosSeq2 = 0;
    
    // 调用条纹 Smith-Waterman 比对函数，在GPU上计算最佳局部比对
    runStripedSW(seq1.c_str(), seq1.length(),
                 seq2.c_str(), seq2.length(),
                 bestScore, endPosSeq1, endPosSeq2);
    
    // 输出比对结果：最佳比对分数以及在序列1和序列2中的结束位置
    std::cout << "Best alignment score: " << bestScore << std::endl;
    std::cout << "Alignment ends at: seq1 index " << endPosSeq1
              << ", seq2 index " << endPosSeq2 << std::endl;
    return 0;
}

