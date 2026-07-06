# ==============================================================================
# 1. Project and Environment Setup
# ==============================================================================
# Define variables (UPDATE THESE FOR YOUR ENVIRONMENT)
set project_name "vivado_build"
set project_dir "./vivado_build"
set ip_repo_path "admm_HW_HLS/solution1/impl/ip"
set admm_vlnv "xilinx.com:hls:admm:1.0" ;# Replace with your actual admm IP VLNV from the catalog

# Create Project for ZCU104
create_project $project_name $project_dir -part xczu7ev-ffvc1156-2-e -force
set_property board_part xilinx.com:zcu104:part0:1.1 [current_project]

# Add IP Repository
set_property ip_repo_paths $ip_repo_path [current_project]
update_ip_catalog

# Create Block Design
create_bd_design "design_1"

# ==============================================================================
# 2. Instantiate Core IP Blocks
# ==============================================================================
# Add Zynq UltraScale+ PS
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ultra_ps_e_0

# Add the custom admm HLS IP
create_bd_cell -type ip -vlnv $admm_vlnv admm_0

# Apply Zynq Board Preset
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1" }  [get_bd_cells zynq_ultra_ps_e_0]

# ==============================================================================
# 3. Configure Zynq PS Interfaces
# ==============================================================================
# Disable HPM1, Enable HPM0, HP0, HP1, HP2, HP3, and PL-PS Interrupts
set_property -dict [list \
  CONFIG.PSU__USE__M_AXI_GP0 {1} \
  CONFIG.PSU__USE__M_AXI_GP1 {0} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} \
  CONFIG.PSU__USE__S_AXI_GP3 {1} \
  CONFIG.PSU__USE__S_AXI_GP4 {1} \
  CONFIG.PSU__USE__S_AXI_GP5 {1} \
  CONFIG.PSU__USE__IRQ0 {1} \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {200} \
] [get_bd_cells zynq_ultra_ps_e_0]

# ==============================================================================
# 4. Connection Automation (AXI SmartConnects, Clocks, and Resets)
# ==============================================================================

# Route the admm Control ports to Zynq HPM0 using AXI SmartConnect
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Master "/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD" Slave "/admm_0/s_axi_control" intc_ip {New AXI SmartConnect} }  [get_bd_intf_pins admm_0/s_axi_control]

# Use Auto for the second control port so Vivado routes it through the SmartConnect we just created
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Master "/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD" Slave "/admm_0/s_axi_control_r" intc_ip {Auto} }  [get_bd_intf_pins admm_0/s_axi_control_r]

# Route the admm GMEM ports to dedicated Zynq HP ports using independent AXI SmartConnects
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Master "/admm_0/m_axi_gmem0" Slave "/zynq_ultra_ps_e_0/S_AXI_HP0_FPD" intc_ip {New AXI SmartConnect} }  [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Master "/admm_0/m_axi_gmem1" Slave "/zynq_ultra_ps_e_0/S_AXI_HP1_FPD" intc_ip {New AXI SmartConnect} }  [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP1_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Master "/admm_0/m_axi_gmem2" Slave "/zynq_ultra_ps_e_0/S_AXI_HP2_FPD" intc_ip {New AXI SmartConnect} }  [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP2_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Master "/admm_0/m_axi_gmem3" Slave "/zynq_ultra_ps_e_0/S_AXI_HP3_FPD" intc_ip {New AXI SmartConnect} }  [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP3_FPD]

# ==============================================================================
# 5. Manual Connections (Interrupts)
# ==============================================================================
# Connect the interrupt line from the admm IP to the Zynq PS
connect_bd_net [get_bd_pins admm_0/interrupt] [get_bd_pins zynq_ultra_ps_e_0/pl_ps_irq0]

# ==============================================================================
# 6. Finalization
# ==============================================================================
# Assign Memory Addresses
assign_bd_address

# Regenerate layout for a clean visual representation
regenerate_bd_layout

# Validate the Block Design
validate_bd_design

# Save the Block Design
save_bd_design

# ==============================================================================
# 7. Generate HDL Wrapper and Set as Top
# ==============================================================================
# Find the block design file
set bd_file [get_files design_1.bd]

# Generate the Verilog/VHDL wrapper
set wrapper_file [make_wrapper -files $bd_file -top]

# Add the wrapper file to the project
add_files -norecurse $wrapper_file

# Set the wrapper as the top-level module
set_property top design_1_wrapper [current_fileset]

# Update compile order to ensure Vivado recognizes the new hierarchy
update_compile_order -fileset sources_1

puts "Block design and HDL Wrapper generated successfully! Project is ready for implementation."