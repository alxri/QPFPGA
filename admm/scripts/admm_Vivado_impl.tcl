# Open the project we just created
open_project vivado_build/vivado_build.xpr
update_compile_order -fileset sources_1

# Tell Vivado to generate a .bin file during the write_bitstream step
set_property STEPS.WRITE_BITSTREAM.ARGS.BIN_FILE true [get_runs impl_1]

# Step 14: Run synthesis, implementation, generate bitstream
launch_runs impl_1 -to_step write_bitstream -jobs 1
wait_on_run impl_1
quit
