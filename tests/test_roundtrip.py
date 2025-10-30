#!/usr/bin/env python3
"""
Roundtrip Test - Verify integrity of an SF2 file

Decompile → compile MuseScore_General_HQ.sf2 and confirm the binary is
identical.

If not identical, verify practical equivalence.
Equivalence checks ignore the following:
- Order of samples, instruments, and presets
- Minor differences in internal sample names and other labels
- Differences in IDs/offsets caused by ordering
"""

import sys
import hashlib
import shutil
from pathlib import Path
from sfutils.decompiler import SF2Decompiler
from sfutils.compiler import SF2Compiler
from test_equivalence import SF2EquivalenceChecker


def calculate_md5(filepath):
    """Calculate the MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def calculate_sha256(filepath):
    """Calculate the SHA256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def main():
    print("=" * 80)
    print("SF2 Roundtrip Test - MuseScore_General_HQ.sf2")
    print("=" * 80)

    # Set file paths
    original_sf2 = Path("MuseScore_General_HQ.sf2")
    temp_dir = Path("temp_roundtrip_test")
    rebuilt_sf2 = Path("temp_rebuilt.sf2")

    # Check that the original file exists
    if not original_sf2.exists():
        print(f"❌ Error: {original_sf2} not found")
        sys.exit(1)

    print(f"\n📁 Original file: {original_sf2}")
    print(f"   Size: {original_sf2.stat().st_size:,} bytes")

    try:
        # Step 1: Decompile
        print(f"\n🔓 Step 1: Decompiling to → {temp_dir}")
        if temp_dir.exists():
            print(f"   Removing existing directory...")
            shutil.rmtree(temp_dir)

        decompiler = SF2Decompiler(str(original_sf2), str(temp_dir))
        decompiler.decompile()
        print(f"   ✓ Decompile complete")

        # Show number of decompiled files
        samples = list((temp_dir / "samples").glob("*.wav"))
        instruments = list((temp_dir / "instruments").glob("*.json"))
        presets = list((temp_dir / "presets").glob("*.json"))
        print(f"   - Samples: {len(samples)}")
        print(f"   - Instruments: {len(instruments)}")
        print(f"   - Presets: {len(presets)}")

        # Step 2: Compile
        print(f"\n🔒 Step 2: Compiling to → {rebuilt_sf2}")
        if rebuilt_sf2.exists():
            print(f"   Removing existing file...")
            rebuilt_sf2.unlink()

        compiler = SF2Compiler(str(temp_dir), str(rebuilt_sf2))
        compiler.compile()
        print(f"   ✓ Compile complete")
        print(f"   Size: {rebuilt_sf2.stat().st_size:,} bytes")

        # Step 3: Binary exact-match check
        print(f"\n🔬 Step 3: Binary exact-match check...")

        original_md5 = calculate_md5(original_sf2)
        rebuilt_md5 = calculate_md5(rebuilt_sf2)

        print(f"   MD5 hashes:")
        print(f"   - Original: {original_md5}")
        print(f"   - Rebuilt:  {rebuilt_md5}")

        is_identical = (original_md5 == rebuilt_md5)

        if is_identical:
            print(f"   ✅ Exact match!")
        else:
            print(f"   ❌ Not identical at the binary level")

        # Step 4: Equivalence check (only if not exactly identical)
        is_equivalent = False
        if not is_identical:
            print(f"\n🔍 Step 4: Equivalence check...")
            print(f"   (Binary exact match failed; checking for practical equivalence)")
            print()

            checker = SF2EquivalenceChecker(str(original_sf2), str(rebuilt_sf2))
            is_equivalent = checker.check()
        else:
            print(f"\n✨ Binary exact match; skipping equivalence check")
            is_equivalent = True

        # Final result
        print("\n" + "=" * 80)
        if is_identical:
            print("🎉 Test passed! Full reversible conversion confirmed!")
            print("   (Binary-level exact match)")
            print("=" * 80)
            success = True
        elif is_equivalent:
            print("✅ Test passed! An equivalent SF2 file was produced")
            print("   (Different binary but practically equivalent)")
            print("=" * 80)
            success = True
        else:
            print("❌ Test failed: Equivalence issues detected")
            print("=" * 80)
            success = False

        # Ask whether to clean up
        print(f"\n🧹 Cleanup:")
        print(f"   Temporary files: {temp_dir}, {rebuilt_sf2}")

        if success:
            response = input("   Delete temporary files? [Y/n]: ").strip().lower()
            if response in ['', 'y', 'yes']:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    print(f"   ✓ Removed {temp_dir}")
                if rebuilt_sf2.exists():
                    rebuilt_sf2.unlink()
                    print(f"   ✓ Removed {rebuilt_sf2}")
            else:
                print(f"   Keeping temporary files")
        else:
            print(f"   Test failed; keeping temporary files for debugging")

        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
