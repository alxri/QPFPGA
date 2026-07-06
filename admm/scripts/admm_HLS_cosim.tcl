# Open the existing project instead of recreating it
open_project admm_HW_HLS
open_solution "solution1"

# Step 1: Synthesize the C++ into RTL hardware (Required before Cosim)
# (If you already clicked C Synthesis in the GUI, you can skip this line)
csynth_design

# Step 2: Run C/RTL Co-simulation
# The -trace_level all flag saves the waveforms so you can view the pipeline execution
cosim_design -trace_level all -tool xsim

quit