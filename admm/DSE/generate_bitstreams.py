import re
import subprocess
import shutil
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_HEADER = REPO_ROOT / "HLS" / "include" / "config.h"
EXPORT_BASE_DIR = REPO_ROOT / "generated_bitstreams"

EXPERIMENTS = {
    "Experiment_1": [
        (1024, 4, 4), (1024, 4, 8), (1024, 4, 16), 
        (1024, 4, 20), (1024, 4, 30), (1024, 4, 40), (1024, 4, 50), (1024, 4, 60)
    ],
    "Experiment_2": [
        (1024, 1, 20), (1024, 2, 20), (1024, 4, 20), (1024, 8, 20)
    ],
    "Experiment_3": [
        (1024, 1, 16), (1024, 1, 20), (1024, 1, 30),
        (1024, 8, 16), (1024, 8, 20), (1024, 8, 30),
        (4096, 1, 16), (4096, 1, 20), (4096, 1, 30),
        (4096, 8, 16), (4096, 8, 20), (4096, 8, 30),
        (8192, 1, 16), (8192, 1, 20), (8192, 1, 30),
        (8192, 8, 16), (8192, 8, 20), (8192, 8, 30),
        (16384, 1, 12), (16384, 1, 16), (16384, 1, 20), (16384, 1, 30),
        (16384, 8, 12), (16384, 8, 16), (16384, 8, 20), (16384, 8, 30),
        (32768, 1, 6), (32768, 1, 8), (32768, 1, 12), (32768, 1, 16), (32768, 1, 20),
        (32768, 8, 6), (32768, 8, 8), (32768, 8, 12), (32768, 8, 16), (32768, 8, 20),
    ]
}

def run_cmd(cmd, description):
    """Run a command, stream output in real-time to avoid pipe deadlocks."""
    print(f"  --> Running: {description}...")
    
    try:
        # Popen allows us to read the output line-by-line as it generates
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge errors into stdout
            text=True,
            bufsize=1 # Line buffered
        )

        # Stream the output directly to the console
        for line in process.stdout:
            # We add a small indent so you know it's from the subprocess
            print(f"      [LOG] {line}", end="")
            
        process.wait()
        
        if process.returncode != 0:
            print(f"FAILED: {description} returned code {process.returncode}")
            return False
        return True
        
    except Exception as e:
        print(f"FAILED (Exception): {description}\n{e}")
        return False

def has_archived_artifacts(target_dir):
    """Check whether a target folder already contains all required outputs."""
    return (
        (target_dir / "admm.bit").exists()
        and (target_dir / "admm.hwh").exists()
        and (target_dir / "vivado_build").exists()
    )

def write_config_header(size, reshape, pes):
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

def archive_artifacts(target_dir, build_success):
    """
    Copy artifacts. If build_success is False, it still tries to 
    copy the vivado_build folder for debugging.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    
    bit_src = REPO_ROOT / "admm.bit"
    hwh_src = REPO_ROOT / "admm.hwh"
    vivado_src = REPO_ROOT / "vivado_build"

    # 1. Copy bitstream files if they exist (usually on success)
    if bit_src.exists():
        shutil.copy2(bit_src, target_dir / "admm.bit")
    if hwh_src.exists():
        shutil.copy2(hwh_src, target_dir / "admm.hwh")

    # 2. Copy the Vivado project directory for debugging (always if it exists)
    if vivado_src.exists():
        vivado_dst = target_dir / "vivado_build"
        if vivado_dst.exists():
            shutil.rmtree(vivado_dst)
        print(f"  --> Archiving vivado_build folder to {target_dir.name}...")
        shutil.copytree(vivado_src, vivado_dst)
    else:
        print(f"  --> Warning: vivado_build folder not found at {vivado_src}")

def main():
    if not EXPORT_BASE_DIR.exists():
        EXPORT_BASE_DIR.mkdir(parents=True)

    for exp_name, configs in EXPERIMENTS.items():
        print(f"\n=== Starting {exp_name} ===")
        
        for size, reshape, pes in configs:
            arch_name = f"{size}x{size}_reshape{reshape}_pes{pes}"
            target_dir = EXPORT_BASE_DIR / arch_name
            
            if target_dir.exists() and has_archived_artifacts(target_dir):
                print(f"Skipping {arch_name}, artifacts already archived.")
                continue

            print(f"\n--- Building Architecture: {arch_name} ---")
            write_config_header(size, reshape, pes)

            build_steps = [
                (["make", "-C", str(REPO_ROOT), "clean"], "Cleaning build directory"),
                (["make", "-C", str(REPO_ROOT), "hls_project"], "Creating HLS Project"),
                (["make", "-C", str(REPO_ROOT), "ip"], "HLS IP Generation"),
                (["make", "-C", str(REPO_ROOT), "vivado_project"], "Creating Vivado Project"),
                (["make", "-C", str(REPO_ROOT), "bitstream"], "Generating Bitstream")
            ]

            overall_success = True
            for cmd, desc in build_steps:
                if not run_cmd(cmd, desc):
                    overall_success = False
                    break # Stop running the next make steps for this architecture
            
            # Archive whatever we have (especially the vivado_build folder)
            archive_artifacts(target_dir, overall_success)

            if overall_success:
                print(f"SUCCESS: {arch_name} completed and archived.")
            else:
                print(f"ERROR: Build failed for {arch_name}. Logs preserved in {target_dir}")
                print("Continuing with remaining bitstreams...")

if __name__ == "__main__":
    main()