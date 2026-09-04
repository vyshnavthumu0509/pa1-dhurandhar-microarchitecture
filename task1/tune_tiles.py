import os
import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CPP_FILE = "src/conv_tile.cpp"
BIN_CMD = ["./bin/conv", "tile", "2048", "2048", "3"]

# Tile sizes to test (powers of 2 are usually best for cache alignment)
TILE_H_VALS = [2, 4, 8, 16, 32, 64]
TILE_W_VALS = [2,4,8,16, 32, 64, 128, 256, 512]

def read_source():
    with open(CPP_FILE, 'r') as f:
        return f.read()

def write_source(content):
    with open(CPP_FILE, 'w') as f:
        f.write(content)

def update_tile_sizes(source, h, w):
    # Regex to find and replace the constexpr definitions
    source = re.sub(r'constexpr\s+int\s+TILE_H\s*=\s*\d+;', f'constexpr int TILE_H = {h};', source)
    source = re.sub(r'constexpr\s+int\s+TILE_W\s*=\s*\d+;', f'constexpr int TILE_W = {w};', source)
    return source

def run_benchmark():
    # Compile the code
    subprocess.run(["make", "clean"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    compile_proc = subprocess.run(["make"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if compile_proc.returncode != 0:
        print("Compilation failed!")
        return 0.0

    # Run the compiled binary testing only the 'tile' stage
    result = subprocess.run(BIN_CMD, capture_output=True, text=True)
    
    # Parse the GFLOP/s from the output
    # The output format has the stage name, correctness, time(ms), GFLOP/s, and speedup
    for line in result.stdout.split('\n'):
        if line.startswith("tile"):
            parts = line.split()
            # If the correctness check failed, return 0 to heavily penalize it
            if parts[1] != "yes":
                return 0.0
            # GFLOP/s is the 4th column (index 3)
            return float(parts[3])
            
    return 0.0

def main():
    original_code = read_source()
    results = np.zeros((len(TILE_H_VALS), len(TILE_W_VALS)))

    print("Starting Tile Size Tuning...")
    try:
        for i, h in enumerate(TILE_H_VALS):
            for j, w in enumerate(TILE_W_VALS):
                print(f"Testing TILE_H={h}, TILE_W={w}...", end=" ", flush=True)
                
                # Modify source
                new_code = update_tile_sizes(original_code, h, w)
                write_source(new_code)
                
                # Run and record
                gflops = run_benchmark()
                results[i, j] = gflops
                print(f"{gflops} GFLOP/s")

    finally:
        # ALWAYS restore the original C++ file, even if the script crashes!
        write_source(original_code)
        print("\nRestored original conv_tile.cpp")

    # --- Plotting the Heatmap ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(results, annot=True, fmt=".2f", 
                xticklabels=TILE_W_VALS, yticklabels=TILE_H_VALS, 
                cmap="viridis", cbar_kws={'label': 'GFLOP/s'})
    
    plt.title("Tiling Performance (GFLOP/s) vs Tile Dimensions")
    plt.xlabel("TILE_W (Width)")
    plt.ylabel("TILE_H (Height)")
    plt.tight_layout()
    plt.savefig("tile_performance.png")
    print("Saved plot to 'tile_performance.png'.")
    plt.show()

if __name__ == "__main__":
    main()