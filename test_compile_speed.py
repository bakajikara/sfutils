#!/usr/bin/env python3
"""
SF2 Compiler Performance Test
"""

import shutil
import os
import time
from pathlib import Path
from sf2_decompiler import SF2Decompiler
from sf2_compiler import SF2Compiler

print("=" * 60)
print("SF2 Compiler Performance Test")
print("=" * 60)

original_sf2 = "MuseScore_General_HQ.sf2"
temp_dir = "temp_speed_test"
output_sf2 = "temp_speed_output.sf2"

# decompile
if not Path(temp_dir).exists():
    print("\n[1/2] Decompiling (one-time setup)...")
    start = time.time()
    decompiler = SF2Decompiler(original_sf2, temp_dir)
    decompiler.decompile()
    elapsed = time.time() - start
    print(f"Decompile time: {elapsed:.2f} seconds")
else:
    print(f"\n[1/2] Using existing directory: {temp_dir}")

# compile (performance test)
print("\n[2/2] Compiling (performance test)...")
start = time.time()
compiler = SF2Compiler(temp_dir, output_sf2)
compiler.compile()
elapsed = time.time() - start

print("\n" + "=" * 60)
print(f"✅ Compilation completed in {elapsed:.2f} seconds")
print("=" * 60)

# file size comparison
orig_size = os.path.getsize(original_sf2)
new_size = os.path.getsize(output_sf2)
print(f"\nFile sizes:")
print(f"  Original: {orig_size:,} bytes")
print(f"  Compiled: {new_size:,} bytes")
print(f"  Match: {orig_size == new_size}")

# clean up
shutil.rmtree(temp_dir)
os.remove(output_sf2)
print("\nTemporary files cleaned up.")
print("\nPerformance test completed.")
