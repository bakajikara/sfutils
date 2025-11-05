#!/usr/bin/env python3
"""
Tool to display sample usage in a SoundFont

Displays samples referenced by each preset in preset number order.
Samples already displayed are not shown again; they appear only under
the first preset that references them.
Finally, displays samples not referenced by any preset.
"""

from sfutils.parser import SoundFontParser
import sys
import os
import argparse

# Add path to import sfutils module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def get_samples_for_instrument(parser, inst_idx, sample_references):
    """
    Get the list of samples used by an instrument.

    Args:
        parser: SoundFontParser instance
        inst_idx: Instrument index
        sample_references: Sample reference information {sample_idx: [ref_info, ...]}

    Returns:
        List of samples [sample_idx, ...]
    """
    zones = parser.get_instrument_zones(inst_idx)
    samples = []

    for zone in zones:
        # Look for sampleID (oper=53) in generators
        for gen in zone["generators"]:
            if gen["oper"] == 53:  # sampleID
                sample_idx = gen["amount"]
                sample_headers = parser.get_sample_headers()
                if 0 <= sample_idx < len(sample_headers):
                    samples.append(sample_idx)

    return samples


def get_samples_for_preset(parser, preset_idx, sample_references):
    """
    Get the list of samples used by a preset.

    Args:
        parser: SoundFontParser instance
        preset_idx: Preset index
        sample_references: Sample reference information {sample_idx: [ref_info, ...]}

    Returns:
        List of samples [sample_idx, ...] (no duplicates)
    """
    zones = parser.get_preset_zones(preset_idx)
    all_samples = []

    for zone in zones:
        # Look for instrument (oper=41) in generators
        for gen in zone["generators"]:
            if gen["oper"] == 41:  # instrument
                inst_idx = gen["amount"]
                samples = get_samples_for_instrument(parser, inst_idx, sample_references)
                all_samples.extend(samples)

    # Remove duplicates while preserving order
    seen = set()
    unique_samples = []
    for sample_idx in all_samples:
        if sample_idx not in seen:
            seen.add(sample_idx)
            unique_samples.append(sample_idx)

    return unique_samples


def show_sample_usage(sf2_path, mode="preset"):
    """
    Display sample usage in a SoundFont.

    Args:
        sf2_path: Path to the SoundFont file
        mode: Display mode ("preset" or "instrument")
    """
    parser = SoundFontParser(sf2_path)
    parser.parse()

    preset_headers = parser.get_preset_headers()
    instrument_headers = parser.get_instrument_headers()
    sample_headers = parser.get_sample_headers()

    # Collect sample reference information
    # sample_references[sample_idx] = [(preset_idx/inst_idx, name), ...]
    sample_references = {}

    print(f"SoundFont: {sf2_path}")
    print(f"Total Presets: {len(preset_headers)}")
    print(f"Total Instruments: {len(instrument_headers)}")
    print(f"Total Samples: {len(sample_headers)}")
    print("=" * 80)
    print()

    if mode == "instrument":
        # Collect reference information for each instrument
        for inst_idx, inst in enumerate(instrument_headers):
            samples = get_samples_for_instrument(parser, inst_idx, sample_references)
            inst_name = inst["name"]

            for sample_idx in samples:
                if sample_idx not in sample_references:
                    sample_references[sample_idx] = []
                sample_references[sample_idx].append((inst_idx, inst_name))

        # Track already displayed samples
        seen_samples = set()

        # Display samples in instrument order
        for inst_idx, inst in enumerate(instrument_headers):
            samples = get_samples_for_instrument(parser, inst_idx, sample_references)

            # Filter for newly found samples only
            new_samples = [s for s in samples if s not in seen_samples]

            if new_samples:
                inst_name = inst["name"]
                print(f"Instrument [{inst_idx:04d}] - {inst_name}")

                for sample_idx in new_samples:
                    sample_name = sample_headers[sample_idx]["name"]
                    refs = sample_references.get(sample_idx, [])

                    # Filter other references (excluding current instrument)
                    other_refs = [ref for ref in refs if ref[0] != inst_idx]

                    if other_refs:
                        other_refs_str = ", ".join([f"{ref[0]:04d}:{ref[1]}" for ref in other_refs])
                        print(f"  [{sample_idx:04d}] {sample_name} (also in: {other_refs_str})")
                    else:
                        print(f"  [{sample_idx:04d}] {sample_name}")

                    seen_samples.add(sample_idx)
                print()
    else:
        # Sort presets by (bank, preset) order
        sorted_presets = sorted(
            enumerate(preset_headers),
            key=lambda x: (x[1]["bank"], x[1]["preset"])
        )

        # Collect reference information for each preset
        for preset_idx, preset in sorted_presets:
            samples = get_samples_for_preset(parser, preset_idx, sample_references)
            bank = preset["bank"]
            preset_num = preset["preset"]
            preset_name = preset["name"]

            # Record presets for each sample (avoid duplicates)
            for sample_idx in samples:
                if sample_idx not in sample_references:
                    sample_references[sample_idx] = []
                # Add (preset_idx, preset_info) tuple (deduplicate later)
                ref_info = (preset_idx, f"{bank:03d}:{preset_num:03d} {preset_name}")
                if ref_info not in sample_references[sample_idx]:
                    sample_references[sample_idx].append(ref_info)

        # Track already displayed samples
        seen_samples = set()

        # Display samples in preset order
        for preset_idx, preset in sorted_presets:
            samples = get_samples_for_preset(parser, preset_idx, sample_references)

            # Filter for newly found samples only
            new_samples = [s for s in samples if s not in seen_samples]

            if new_samples:
                bank = preset["bank"]
                preset_num = preset["preset"]
                preset_name = preset["name"]

                print(f"Preset {bank:03d}:{preset_num:03d} - {preset_name}")

                for sample_idx in new_samples:
                    sample_name = sample_headers[sample_idx]["name"]
                    refs = sample_references.get(sample_idx, [])

                    # Filter other references (excluding current preset)
                    other_refs = [ref for ref in refs if ref[0] != preset_idx]

                    if other_refs:
                        other_refs_str = ", ".join([ref[1] for ref in other_refs])
                        print(f"  [{sample_idx:04d}] {sample_name} (also in: {other_refs_str})")
                    else:
                        print(f"  [{sample_idx:04d}] {sample_name}")

                    seen_samples.add(sample_idx)
                print()    # Display unused samples
    all_sample_indices = set(range(len(sample_headers)))
    unused_samples = all_sample_indices - seen_samples

    if unused_samples:
        print("=" * 80)
        print("Unused Samples:")
        print("=" * 80)
        for sample_idx in sorted(unused_samples):
            sample_name = sample_headers[sample_idx]["name"]
            sample_type = sample_headers[sample_idx]["sample_type"]
            print(f"  [{sample_idx:04d}] {sample_name} (type={sample_type})")
        print()
        print(f"Unused sample count: {len(unused_samples)}")
    else:
        print("=" * 80)
        print("All samples are used.")
        print("=" * 80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Display sample usage in a SoundFont.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display by preset (default)
  python show_sample_usage.py soundfont.sf2

  # Display by instrument
  python show_sample_usage.py soundfont.sf2 --mode instrument
  python show_sample_usage.py soundfont.sf2 -m i
        """
    )

    parser.add_argument(
        "sf2_file",
        help="Path to the SoundFont file"
    )

    parser.add_argument(
        "-m", "--mode",
        choices=["preset", "instrument", "p", "i"],
        default="preset",
        help="Display mode: preset (p) = by preset, instrument (i) = by instrument (default: preset)"
    )

    args = parser.parse_args()

    # Convert mode abbreviations to full names
    mode = args.mode
    if mode == "p":
        mode = "preset"
    elif mode == "i":
        mode = "instrument"

    sf2_path = args.sf2_file

    if not os.path.exists(sf2_path):
        print(f"Error: File not found: {sf2_path}")
        sys.exit(1)

    try:
        show_sample_usage(sf2_path, mode)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
