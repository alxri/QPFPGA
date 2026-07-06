.PHONY: hls_project hls_sim hls_synth ip vivado_project bitstream clean cleanall help
PROJECT_NAME := admm

help:
	@echo "VITIS HLS targets"
	@echo "  hls_project:    Creates the Vitis HLS project" 
	@echo "  hls_sim:        Runs C simulation" 
	@echo "  hls_synth:      Runs C-to-RTL synthesis"
	@echo "  ip:             Exports the synthesized design as an IP core" 
	@echo ""
	@echo "VIVADO targets"
	@echo "  vivado_project: Creates Vivado project in vivado_build/ and builds BD" 
	@echo "  bitstream:      Generates bitstream and hardware handoff" 

# --- VITIS HLS ---

# 1. Setup the project
hls_project:
	rm -rf $(PROJECT_NAME)_HW_HLS
	vitis_hls -f scripts/$(PROJECT_NAME)_HLS.tcl

# 2. Run Synthesis (CSYNTH)
hls_synth:
	vitis_hls -f scripts/$(PROJECT_NAME)_HLS_csynth.tcl
	cp admm_HW_HLS/solution1/syn/report/admm_csynth.rpt reports/

# 3. Run Simulation (CSIM)
hls_sim:
	vitis_hls -f scripts/$(PROJECT_NAME)_HLS_sim.tcl

hls_cosim:
	vitis_hls -f scripts/$(PROJECT_NAME)_HLS_cosim.tcl
	cp admm_HW_HLS/solution1/syn/report/admm_csynth.rpt reports/
	cp admm_HW_HLS/solution1/sim/report/admm_cosim.rpt reports/

# 4. Packaging (EXPORT)
ip: hls_synth
	vitis_hls -f scripts/$(PROJECT_NAME)_HLS_export.tcl
	cp admm_HW_HLS/solution1/syn/report/admm_csynth.rpt reports/


# --- VIVADO ---

# Create the Vivado project and Block Design (Depends on ip)
vivado_project:
	rm -rf vivado_build
	mkdir -p vivado_build
	vivado -mode batch -source scripts/$(PROJECT_NAME)_Vivado.tcl

# Execute synthesis, implementation and bitstream generation 
bitstream:
	vivado -mode batch -source scripts/$(PROJECT_NAME)_Vivado_impl.tcl
	cp ./vivado_build/vivado_build.runs/impl_1/design_1_wrapper.bit ./$(PROJECT_NAME).bit
	cp ./vivado_build/vivado_build.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh ./$(PROJECT_NAME).hwh

clean:
	rm -rf .Xil vivado*.jou vivado*.log vivado_build admm_HW_HLS IP-catalog/*.zip *.ila
