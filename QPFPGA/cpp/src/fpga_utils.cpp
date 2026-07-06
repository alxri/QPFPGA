#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "utils.h"
#include "fpga_utils.h"

// PYNQ libcma exports
extern "C" {
    void* cma_alloc(uint32_t len, uint32_t cacheable);
    unsigned long cma_get_phy_addr(void *buf);
    void cma_free(void *buf);
    void cma_flush_cache(void *buf, unsigned long phys_addr, int size);
    void cma_invalidate_cache(void *buf, unsigned long phys_addr, int size);
}

namespace fs = std::filesystem;

// path: bitstreams/*.bit

bool load_bitstream(const std::string& path) {
    fs::path src(path);
    if (!fs::exists(src)) {
        std::cerr << "Error: Bitstream file does not exist at " << path << "\n";
        return false;
    }

    // Copy the bitstream to /lib/firmware
    fs::path dest_dir = "/lib/firmware";
    fs::path dest = dest_dir / src.filename();
    
    try {
        // Overwrite if it already exists in /lib/firmware
        fs::copy_file(src, dest, fs::copy_options::overwrite_existing);
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error copying bitstream to /lib/firmware: " << e.what() << "\n";
        std::cerr << "Are you running as root/sudo?\n";
        return false;
    }

    // Set FPGA Manager flags (0 = full bitstream)
    std::ofstream flags_file("/sys/class/fpga_manager/fpga0/flags");
    if (flags_file) {
        flags_file << "0";
        flags_file.close();
    } else {
        std::cerr << "Warning: Could not open FPGA manager flags. Continuing anyway.\n";
    }

    // Write filename to the firmware node to trigger programming
    std::ofstream fw_file("/sys/class/fpga_manager/fpga0/firmware");
    if (!fw_file) {
        std::cerr << "Error: Could not open FPGA manager firmware node.\n";
        return false;
    }
    fw_file << src.filename().string();
    fw_file.close();

    // Verify programming was successful
    std::ifstream state_file("/sys/class/fpga_manager/fpga0/state");
    std::string state;
    if (state_file) {
        state_file >> state;
        if (state != "operating") {
            std::cerr << "Error: FPGA programming failed. State is: " << state << "\n";
            return false;
        }
    }

    std::cout << "Bitstream loaded successfully.\n";
    return true;
}

void write_reg(void *base, uint32_t offset, uint32_t val) { *((volatile uint32_t *)((uint8_t *)base + offset)) = val; }
uint32_t read_reg(void *base, uint32_t offset) { return *((volatile uint32_t *)((uint8_t *)base + offset)); }
void write_64bit_address(void *base, uint32_t offset, uintptr_t address) {
    write_reg(base, offset, (uint32_t)(address & 0xFFFFFFFF));
    write_reg(base, offset + 0x04, (uint32_t)((uint64_t)address >> 32));
}

// Program a list of (offset, physical_address) into the control_r registers
void program_cma_addresses(void* ctrl_r, const std::vector<std::pair<uint32_t, uintptr_t>>& addr_list) {
    for (const auto &p : addr_list) {
        write_64bit_address(ctrl_r, p.first, p.second);
    }
}



// Allocate CMA buffers for CSC arrays and copy packed-word data into them.
void allocate_and_copy_csc_to_cma(
    CmaTracker& cma,
    const std::vector<int>& cptr,
    const std::vector<int>& ridx,
    const std::vector<float>& vals,
    int32_words** out_ridx_reg,
    float32_words** out_vals_reg,
    int** out_cptr_reg,
    int* out_words_cnt)
{
    int words_cnt = ceil_div((int)vals.size(), PACK_SIZE);
    // allocate CMA buffers
    int32_words* reg_ridx = cma.alloc<int32_words>(words_cnt);
    float32_words* reg_vals = cma.alloc<float32_words>(words_cnt);
    int* reg_cptr = cma.alloc<int>(cptr.size());

    // Pack into temporary word arrays then copy
    std::vector<int32_words> temp_ridx;
    std::vector<float32_words> temp_vals;
    pack_indices_to_words(ridx, temp_ridx);
    pack_vals_to_words(vals, temp_vals);

    cma_copy(reg_cptr, cptr.data(), (size_t)cptr.size());
    cma_copy(reg_ridx, temp_ridx.data(), (size_t)words_cnt);
    cma_copy(reg_vals, temp_vals.data(), (size_t)words_cnt);

    *out_ridx_reg = reg_ridx;
    *out_vals_reg = reg_vals;
    *out_cptr_reg = reg_cptr;
    *out_words_cnt = words_cnt;
}