#!/usr/bin/env python3
"""
Roundtrip Test - Verify integrity of SoundFont files

Decompiles and recompiles SoundFont files under different conditions
to confirm the process is reversible or produces an equivalent file.

Usage:
  python test_roundtrip.py          (Runs all tests)
  python test_roundtrip.py 1 3      (Runs tests #1 and #3 only)
"""

import sys
import hashlib
import shutil
import argparse  # Import the argparse library
from pathlib import Path
from sfutils import SoundFontCompiler, SoundFontDecompiler
from test_equivalence import SoundFontEquivalenceChecker


def calculate_md5(filepath):
    """Calculate the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def run_test_case(test_name: str, original_file: Path, decompiler_options: dict, compiler_options: dict) -> bool:
    """
    Runs a single decompile-compile roundtrip test case.
    """
    print("\n" + "=" * 80)
    print(f"▶️  Running Test Case: {test_name}")
    print("=" * 80)

    safe_test_name = "".join(c if c.isalnum() else "_" for c in test_name)
    temp_dir = Path(f"temp_roundtrip_{safe_test_name}")
    rebuilt_file = Path(f"temp_rebuilt_{safe_test_name}{original_file.suffix}")

    if not original_file.exists():
        print(f"❌ Error: Original file not found at \"{original_file}\"")
        print("   Skipping this test case.")
        return False

    print(f"📁 Original file: {original_file} ({original_file.stat().st_size:,} bytes)")
    success = False

    try:
        print(f"\n🔓 Step 1: Decompiling to → {temp_dir}")
        if temp_dir.exists():
            print(f"   Removing existing directory...")
            shutil.rmtree(temp_dir)

        decompiler = SoundFontDecompiler(str(original_file), str(temp_dir), **decompiler_options)
        decompiler.decompile()
        print("   ✓ Decompile complete")

        print(f"\n🔒 Step 2: Compiling to → {rebuilt_file}")
        if rebuilt_file.exists():
            rebuilt_file.unlink()

        compiler = SoundFontCompiler(str(temp_dir), str(rebuilt_file), **compiler_options)
        compiler.compile()
        print("   ✓ Compile complete")
        print(f"   Rebuilt size: {rebuilt_file.stat().st_size:,} bytes")

        print("\n🔬 Step 3: Binary exact-match check...")
        is_identical = False
        is_equivalent = False

        original_md5 = calculate_md5(original_file)
        rebuilt_md5 = calculate_md5(rebuilt_file)

        print(f"   MD5 (Original): {original_md5}")
        print(f"   MD5 (Rebuilt):  {rebuilt_md5}")

        is_identical = (original_md5 == rebuilt_md5)

        if is_identical:
            print("   ✅ Exact binary match!")
            is_equivalent = True
        else:
            print("   ❌ Not identical at the binary level.")
            print("\n🔍 Step 4: Equivalence check...")
            checker = SoundFontEquivalenceChecker(str(original_file), str(rebuilt_file))
            is_equivalent = checker.check()

        print("\n" + "-" * 40)
        if is_equivalent:
            print(f"PASS: {test_name}")
            success = True
        else:
            print(f"FAIL: {test_name}")
            success = False
        print("-" * 40)

    except Exception as e:
        print(f"\n❌ An unexpected error occurred during \"{test_name}\": {e}")
        import traceback
        traceback.print_exc()
        success = False

    finally:
        print("\n🧹 Cleanup...")
        if success:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                print(f"   ✓ Removed {temp_dir}")
            if rebuilt_file.exists():
                rebuilt_file.unlink()
                print(f"   ✓ Removed {rebuilt_file}")
        else:
            print(f"   ⚠️ Test failed; keeping temporary files for debugging:")
            print(f"      - Decompiled files: {temp_dir}")
            print(f"      - Rebuilt file: {rebuilt_file}")

    return success


def main():
    """Defines and runs all roundtrip test cases."""

    # Define all available test cases
    test_cases = [
        {
            "test_name": "SF2 Default Roundtrip",
            "original_file": Path("MuseScore_General_HQ.sf2"),
            "decompiler_options": {"split_stereo": False},
            "compiler_options": {},
        },
        {
            "test_name": "SF2 Split-Stereo Roundtrip",
            "original_file": Path("MuseScore_General_HQ.sf2"),
            "decompiler_options": {"split_stereo": True},
            "compiler_options": {},
        },
        {
            "test_name": "SF3 Equivalence Roundtrip",
            "original_file": Path("MuseScore_General_HQ.sf3"),
            "decompiler_options": {},
            "compiler_options": {},
        },
    ]

    parser = argparse.ArgumentParser(description="Run SoundFont roundtrip tests.")
    parser.add_argument(
        "test_numbers",
        nargs="*",
        type=int,
        help="Optional numbers of the specific tests to run (e.g., 1 3). If omitted, all tests are run."
    )
    args = parser.parse_args()

    print("Available Test Cases:")
    for i, case in enumerate(test_cases, 1):
        print(f"  {i}: {case["test_name"]}")

    tests_to_run = []
    if not args.test_numbers:
        # If no numbers are provided, select all tests
        tests_to_run = test_cases
        print("\nNo specific tests selected. Running all tests.")
    else:
        # If numbers are provided, select only those tests
        valid_numbers = range(1, len(test_cases) + 1)
        for num in args.test_numbers:
            if num not in valid_numbers:
                print(f"\n❌ Error: Invalid test number \"{num}\". Please choose from the list above.")
                sys.exit(1)
            tests_to_run.append(test_cases[num - 1])  # Adjust for 0-based index

        selected_numbers_str = ", ".join(map(str, args.test_numbers))
        print(f"\nRunning selected tests: #{selected_numbers_str}")

    # --- Run the selected tests ---
    results = {}
    for case in tests_to_run:
        decompiler_opts = case.get("decompiler_options", {})
        compiler_opts = case.get("compiler_options", {})

        result = run_test_case(
            case["test_name"],
            case["original_file"],
            decompiler_opts,
            compiler_opts,
        )
        results[case["test_name"]] = result

    # --- Final Summary ---
    print("\n" + "=" * 80)
    print("⏹️  Roundtrip Test Suite Summary")
    print("=" * 80)

    # Check if any tests were actually run
    if not results:
        print("No tests were run.")
        sys.exit(0)

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status.ljust(8)} | {name}")
        if not passed:
            all_passed = False

    print("-" * 80)
    if all_passed:
        print("🎉 All selected test cases passed!")
        sys.exit(0)
    else:
        print("🔥 Some selected test cases failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
