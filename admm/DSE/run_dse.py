import os
import subprocess
import itertools
import csv
import re
import time
import shutil
from pathlib import Path

# ==========================================
# 1. DEFINE EXPLORATION SPACE
# ==========================================
# NUM_PES_LIST = [4, 8, 16, 20, 30, 40, 50, 60]
# SIZES_LIST = [512, 1024, 4096, 8192, 16384, 32768] # Assumes square (NUM_COLS = NUM_ROWS)
# RESHAPE_FACTORS = [2, 4, 8]

NUM_PES_LIST = [6]
SIZES_LIST = [32768] # Assumes square (NUM_COLS = NUM_ROWS)
RESHAPE_FACTORS = [8]

# ==========================================
# 2. CONFIGURATION & PATHS
# ==========================================
REPO_ROOT = Path(__file__).resolve().parents[1]
HLS_PROJECT_DIR = REPO_ROOT / "admm_HW_HLS"
SOLUTION_DIR = "solution1"
TOP_FUNCTION = "admm"

CONFIG_HEADER = REPO_ROOT / "HLS" / "include" / "config.h"
OUTPUT_CSV = REPO_ROOT / "DSE" / "dse_results_csynth_only_added.csv" # Renamed to avoid overwriting later cosim results
OUTPUT_TABLE_MD = REPO_ROOT / "DSE" / "dse_results_csynth_only.md"

SYNTH_REPORT = HLS_PROJECT_DIR / SOLUTION_DIR / "syn" / "report" / f"{TOP_FUNCTION}_csynth.rpt"
COSIM_REPORT = HLS_PROJECT_DIR / SOLUTION_DIR / "sim" / "report" / f"{TOP_FUNCTION}_cosim.rpt"

# ==========================================
# 3. PARSING FUNCTIONS
# ==========================================
def parse_csynth_report(filepath):
    """Extracts BRAM, DSP, FF, LUT, and URAM utilization percentages from C-Synthesis report."""
    utilization = {"BRAM": "N/A", "DSP": "N/A", "FF": "N/A", "LUT": "N/A", "URAM": "N/A"}
    if not os.path.exists(filepath):
        return utilization

    with open(filepath, 'r') as f:
        content = f.read()

        # Compute utilization from the Total and Available rows to avoid summary-row inconsistencies.
        total_match = re.search(r'\|\s*Total\s*\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|', content)
        available_match = re.search(r'\|\s*Available\s*\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|', content)
        if total_match and available_match:
            total_values = [int(total_match.group(index)) for index in range(1, 6)]
            available_values = [int(available_match.group(index)) for index in range(1, 6)]

            keys = ["BRAM", "DSP", "FF", "LUT", "URAM"]
            for key, total_value, available_value in zip(keys, total_values, available_values):
                if available_value > 0:
                    utilization[key] = str(round((total_value / available_value) * 100))
            return utilization

        # Fall back to the report's explicit utilization row when totals are unavailable.
        match = re.search(r'\|\s*Utilization\s*\(%\)\s*\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|', content)
        if match:
            utilization["BRAM"] = match.group(1)
            utilization["DSP"] = match.group(2)
            utilization["FF"] = match.group(3)
            utilization["LUT"] = match.group(4)
            utilization["URAM"] = match.group(5)
            
    return utilization

def parse_cosim_report(filepath):
    """Extracts latency cycles from the Cosimulation report."""
    latency = "N/A" # Default to N/A for synth-only runs
    if not os.path.exists(filepath):
        return latency

    with open(filepath, 'r') as f:
        content = f.read()
        # Prefer the Verilog Pass row because it contains the completed RTL run.
        match = re.search(
            r'^\|\s*Verilog\s*\|\s*Pass\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|.*\|\s*(\d+)\s*\|$',
            content,
            re.MULTILINE,
        )
        if match:
            latency = match.group(2)
        else:
            match = re.search(
                r'^\|\s*VHDL\s*\|\s*Pass\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|.*\|\s*(\d+)\s*\|$',
                content,
                re.MULTILINE,
            )
            if match:
                latency = match.group(2)
            
    return latency

def render_markdown_table(rows, headers):
    """Render rows as a compact markdown table."""
    table_lines = []
    table_lines.append("| " + " | ".join(headers) + " |")
    table_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        table_lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(table_lines) + "\n"

def update_config_header(size, reshape, pes):
    """Update only the architecture-specific fields in HLS/include/config.h."""
    content = CONFIG_HEADER.read_text()

    replacements = [
        (r"(?m)^(#define\s+NUM_PES\s+)\S+", rf"\g<1>{pes}"),
        (r"(?m)^(#define\s+MAX_ROWS\s+)\S+", rf"\g<1>{size}"),
        (r"(?m)^(#define\s+MAX_COLS\s+)\S+", rf"\g<1>{size}"),
        (r"(?m)^(#define\s+RESHAPE_FACTOR\s+)\S+", rf"\g<1>{reshape}"),
        (r"(?m)^(#define\s+MAX_SIZE\s+)\S+", rf"\g<1>{size}"),
    ]

    for pattern, replacement in replacements:
        content, count = re.subn(pattern, replacement, content, count=1)
        if count != 1:
            raise RuntimeError(f"Failed to update config header field for pattern: {pattern}")

    CONFIG_HEADER.write_text(content)

# ==========================================
# 4. MAIN DSE LOOP
# ==========================================
def main():
    if shutil.which("vitis_hls") is None:
        raise SystemExit(
            "vitis_hls was not found on PATH. Source the Vitis/Vivado settings script first, then rerun this DSE script."
        )

    combinations = list(itertools.product(NUM_PES_LIST, SIZES_LIST, RESHAPE_FACTORS))
    total_runs = len(combinations)
    
    print(f"Starting C-Synthesis Design Space Exploration for {total_runs} combinations...")
    
    results = []

    # Setup CSV Writer
    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Num_PEs", "Size", "Reshape_Factor", "Latency_Cycles", "BRAM_%", "DSP_%", "FF_%", "LUT_%", "URAM_%", "Status"])
        
        for idx, (pes, size, reshape) in enumerate(combinations):
            print(f"\n[{idx+1}/{total_runs}] C-Synth -> PEs: {pes}, Size: {size}x{size}, Reshape: {reshape}")
            
            # Step 1: Write header file
            update_config_header(size, reshape, pes)
                
            # Step 2: Clean previous reports to avoid reading stale data
            if os.path.exists(SYNTH_REPORT): os.remove(SYNTH_REPORT)
            if os.path.exists(COSIM_REPORT): os.remove(COSIM_REPORT)
                
            # Step 3: Run Make (CHANGED to a synthesis-only target)
            start_time = time.time()
            try:
                # Ensure "hls_csynth" matches the target name in your Makefile
                # If your Makefile cleans automatically, great. If not, you might need a clean target first.
                process = subprocess.run(["make", "-C", str(REPO_ROOT), "hls_synth"], capture_output=True, text=True, timeout=1200) # Reduced timeout for synth
                
                if process.returncode != 0:
                    print("  -> Synthesis Failed (Compilation or HLS routing error).")
                    if process.stdout:
                        print(process.stdout.rstrip()[-500:]) # Print only the tail of the error to avoid flooding the console
                    status = "Failed"
                else:
                    status = "Success"
                    
            except subprocess.TimeoutExpired:
                print("  -> Synthesis Timed Out.")
                status = "Timeout"
                
            elapsed = time.time() - start_time
            print(f"  -> Finished in {elapsed:.1f} seconds. Status: {status}")
            
            # Step 4: Parse Reports
            utilization = parse_csynth_report(SYNTH_REPORT)
            latency = parse_cosim_report(COSIM_REPORT) # Will safely return "N/A"
            
            # Step 5: Save Results
            row = [
                pes, size, reshape, 
                latency, 
                utilization["BRAM"], utilization["DSP"], utilization["FF"], utilization["LUT"], utilization["URAM"],
                status
            ]
            writer.writerow(row)
            results.append(row)
            file.flush() # Ensure data is written immediately

    table_headers = ["Num_PEs", "Size", "Reshape", "BRAM_%", "DSP_%", "FF_%", "LUT_%", "URAM_%", "Status"]
    table_rows = [
        [pes, size, reshape, bram, dsp, ff, lut, uram, status]
        for pes, size, reshape, _, bram, dsp, ff, lut, uram, status in results
    ]
    OUTPUT_TABLE_MD.write_text(render_markdown_table(table_rows, table_headers))

    print(f"\nC-Synthesis Exploration Complete! Results saved to {OUTPUT_CSV}")
    print(f"Markdown table saved to {OUTPUT_TABLE_MD}")

if __name__ == "__main__":
    main()