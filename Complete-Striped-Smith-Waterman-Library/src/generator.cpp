#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <cstdlib>

// 生成隨機 A/T/C/G DNA 序列
std::string generate_random_dna(int length) {
    static const char bases[] = { 'A', 'T', 'C', 'G' };
    std::string sequence;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 3);

    for (int i = 0; i < length; ++i) {
        sequence += bases[dist(gen)];
    }

    return sequence;
}

// 寫入 fasta 檔案
void write_fasta(const std::string& filename, const std::string& header, const std::string& sequence) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    out << ">" << header << "\n";
    for (size_t i = 0; i < sequence.size(); i += 60) {
        out << sequence.substr(i, 60) << "\n";
    }
    out.close();
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <query_length> <ref_length>\n";
        return 1;
    }

    int query_len = std::atoi(argv[1]);
    int ref_len = std::atoi(argv[2]);

    if (query_len <= 0 || ref_len <= 0) {
        std::cerr << "Lengths must be positive integers.\n";
        return 1;
    }

    std::string query_seq = generate_random_dna(query_len);
    std::string ref_seq = generate_random_dna(ref_len);

    write_fasta("query.fasta", "random_query", query_seq);
    write_fasta("ref.fasta", "random_ref", ref_seq);

    std::cout << "Generated query.fasta (" << query_len << "bp) and ref.fasta (" << ref_len << "bp).\n";

    return 0;
}
