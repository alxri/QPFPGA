open_project admm_HW_HLS
set_top admm

# Design Files
add_files HLS/include/spmv_csc.h
add_files HLS/include/admm.h
add_files HLS/src/spmv_csc.cpp -cflags "-I HLS/include"
add_files HLS/src/pcg.cpp -cflags "-I HLS/include"
add_files HLS/src/admm.cpp -cflags "-I HLS/include"



# Testbench Files - Adding it here makes it available for hls_sim
add_files -tb HLS/test/tb_admm.cpp -cflags "-I HLS/include -I HLS/test"

open_solution "solution1" -flow_target vivado
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 5 -name default
set_clock_uncertainty 1
config_interface -m_axi_addr64=1
quit