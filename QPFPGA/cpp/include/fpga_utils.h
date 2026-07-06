#ifndef FPGA_UTILS_H
#define FPGA_UTILS_H

#include <cstdint>
#include <vector>
#include <utility>
#include "utils.h"

void write_reg(void *base, uint32_t offset, uint32_t val);
uint32_t read_reg(void *base, uint32_t offset);
void write_64bit_address(void *base, uint32_t offset, uintptr_t address);

int load_bitstream(const char* path);

void allocate_and_copy_csc_to_cma(
    CmaTracker& cma,
    const std::vector<int>& cptr,
    const std::vector<int>& ridx,
    const std::vector<float>& vals,
    int32_words** out_ridx_reg,
    float32_words** out_vals_reg,
    int** out_cptr_reg,
    int* out_words_cnt);

// Helper to program many 64-bit CMA addresses into control_r registers
void program_cma_addresses(void* ctrl_r, const std::vector<std::pair<uint32_t, uintptr_t>>& addr_list);


#endif // FPGA_UTILS_H