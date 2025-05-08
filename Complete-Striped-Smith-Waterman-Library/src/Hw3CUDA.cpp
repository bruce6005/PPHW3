#include <cuda_runtime.h>
#include "HW2_cuda.cuh"
#include <iostream>
#include <string.h>
#include <cstring>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "ssw_cpp.h"
#include <cstdint>




using std::string;
using std::cout;
using std::endl;
using namespace std;
using namespace chrono;
using std::vector;

struct AlignmentResult {
    int score;
    int ref_end;
    int query_end;
    vector<std::vector<int>> DP;
};



string read_fasta(const string& path) {
    ifstream file(path);
    string line, seq;
    while (getline(file, line)) {
        if (line.empty()) continue;
        if (line[0] == '>') continue;
        seq += line;
    }
    return seq;
}

AlignmentResult striped_sw_cuda(const std::string& ref, const std::string& query,
    int match = 2, int mismatch = -1, int gap = -2) {
        int refLen = ref.size();
        int queryLen = query.size();
        std::vector<std::vector<int>> DP(refLen + 1, std::vector<int>(queryLen + 1, 0));
        std::vector<int> DP_flat((refLen + 1) * (queryLen + 1), 0);

        striped_sw_cuda_launcher(ref.data(), refLen, query.data(), queryLen,
        match, mismatch, gap, DP_flat.data());

        for (int i = 0; i <= refLen; ++i)
            for (int j = 0; j <= queryLen; ++j)
                DP[i][j] = DP_flat[i * (queryLen + 1) + j];

        int max_score = 0, end_ref = -1, end_query = -1;
        for (int i = 1; i <= refLen; ++i)
            for (int j = 1; j <= queryLen; ++j)
                if (DP[i][j] > max_score) {
        max_score = DP[i][j];
        end_ref = i - 1;
        end_query = j - 1;
    }

    return {max_score, end_ref, end_query, DP};
}

void print_dp_matrix(const std::vector<std::vector<int>>& DP,
    const std::string& ref, const std::string& query) {
    int refLen = ref.size();
    int queryLen = query.size();

    // Column label row
    std::cout << std::setw(6) << ' ';
    for (int j = 0; j <= queryLen; ++j) {
    if (j == 0)
    std::cout << std::setw(4) << '-';
    else
    std::cout << std::setw(4) << query[j - 1];
    }
    std::cout << '\n';

    // DP rows with ref label
    for (int i = 0; i <= refLen; ++i) {
    if (i == 0)
    std::cout << std::setw(4) << '-' << " ";
    else
    std::cout << std::setw(4) << ref[i - 1] << " ";

    for (int j = 0; j <= queryLen; ++j) {
    std::cout << std::setw(4) << DP[i][j];
    }
    std::cout << '\n';
    }
}

int main(){
    string ref = read_fasta("ref.fasta");
    // int refSize= ref.size();
    string query = read_fasta("query.fasta");
    cout<< "ql :"<<query.length()<<endl;
    auto start = high_resolution_clock::now();
    AlignmentResult result = striped_sw_cuda(ref, query);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(end - start);

    print_dp_matrix(result.DP, ref, query);
    cout<<"query_end: "<<result.query_end<<" "<<"ref_end: "<<result.ref_end<<endl;
    cout<<"CUDA: "<<duration.count()<<" sec"<<endl;
    return 0;

}