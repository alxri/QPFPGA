#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint> // Required for int32_t

// Helper function to read a binary file into a vector
template <typename T>
std::vector<T> read_binary_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open file " << filepath << std::endl;
        return std::vector<T>();
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<T> buffer(size / sizeof(T));
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return buffer;
    }

    return std::vector<T>();
}

// Helper function to print a vector
template <typename T>
void print_vector(const std::string& name, const std::vector<T>& vec) {
    std::cout << "=== " << name << " (Size: " << vec.size() << ") ===\n";
    if (vec.empty()) {
        std::cout << "[Empty or Not Found]\n\n";
        return;
    }
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << " ";
        if ((i + 1) % 10 == 0) std::cout << "\n"; // Wrap lines for readability
    }
    std::cout << "\n\n";
}

int main() {
    // Relative path to your data directory
    const std::string path = "data_bin/";

    // 1. Load Vectors (np.float32 -> float)
    std::vector<float> l = read_binary_file<float>(path + "l.bin");
    std::vector<float> q = read_binary_file<float>(path + "q.bin");
    std::vector<float> u = read_binary_file<float>(path + "u.bin");
    std::vector<float> P_diag = read_binary_file<float>(path + "P_diag.bin");

    // 2. Load CSC Matrix Components 
    // Data: np.float32 -> float
    // Indices: np.int32 -> int32_t
    std::vector<float> A_data = read_binary_file<float>(path + "A_data.bin");
    std::vector<int32_t> A_indices = read_binary_file<int32_t>(path + "A_indices.bin");
    std::vector<int32_t> A_indptr  = read_binary_file<int32_t>(path + "A_indptr.bin");

    std::vector<float> P_data = read_binary_file<float>(path + "P_data.bin");
    std::vector<int32_t> P_indices = read_binary_file<int32_t>(path + "P_indices.bin");
    std::vector<int32_t> P_indptr  = read_binary_file<int32_t>(path + "P_indptr.bin");

    // 3. Print Dense Vectors
    print_vector("Vector l", l);
    print_vector("Vector q", q);
    print_vector("Vector u", u);
    print_vector("P_diag", P_diag);

    // 4. Print CSC Matrix Components
    print_vector("A_data", A_data);
    print_vector("A_indices", A_indices);
    print_vector("A_indptr", A_indptr);

    print_vector("P_data", P_data);
    print_vector("P_indices", P_indices);
    print_vector("P_indptr", P_indptr);

    return 0;
}