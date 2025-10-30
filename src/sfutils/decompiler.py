#!/usr/bin/env python3
"""
SF2 Decompiler - Decompiles a SoundFont2 file into a directory structure.

This tool expands an SF2 file into the following structure:
- info.json: Metadata
- samples/: FLAC files (waveform data)
- instruments/: Instrument definitions (JSON)
- presets/: Preset definitions (JSON)
"""

import os
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import soundfile as sf
except ImportError:
    print("Error: soundfile library is required. Install it with: pip install soundfile")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: numpy library is required. Install it with: pip install numpy")
    sys.exit(1)

from sfutils.constants import GENERATOR_NAMES
from sfutils.parser import SF2Parser


def sanitize_filename(name):
    """
    Replaces characters that are invalid in filenames with underscores.

    Args:
        name: The original filename.

    Returns:
        The sanitized filename.
    """
    invalid_chars = "<>:\"/\\|?*"
    for char in invalid_chars:
        name = name.replace(char, "_")
    return name.strip()


class SF2Decompiler:
    """
    A class to decompile an SF2 file into a directory structure.
    """

    def __init__(self, sf2_path, output_dir):
        """
        Initializes the SF2Decompiler.

        Args:
            sf2_path: The path to the SF2 file.
            output_dir: The output directory path.
        """
        self.sf2_path = sf2_path
        self.output_dir = Path(output_dir)
        self.parser = SF2Parser(sf2_path)
        self.sample_id_to_filename = {}

    def decompile(self):
        """
        Decompiles the SF2 file.
        """
        print(f"Parsing SF2 file: {self.sf2_path}")
        self.parser.parse()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Decompiling to: {self.output_dir}")

        # Decompile each part
        self._export_bank_info()
        self._export_samples()
        self._export_instruments()
        self._export_presets()

        print("Decompilation complete!")

    def _export_bank_info(self):
        """
        Exports info.json.
        """
        output_path = self.output_dir / "info.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.parser.info_data, f, indent=2, ensure_ascii=False)
        print(f"  Created: info.json")

    def _export_samples(self):
        """
        Exports FLAC files and metadata to the samples directory.
        """
        samples_dir = self.output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        sample_headers = self.parser.get_sample_headers()
        sample_data = self.parser.sample_data

        # Prepare tasks for parallel processing
        tasks = self._prepare_sample_tasks(sample_headers)

        # Process tasks in parallel
        results = self._process_sample_tasks_parallel(tasks, samples_dir, sample_data, sample_headers)

        # Update filename mappings
        for result in results:
            self.sample_id_to_filename.update(result["id_mapping"])

        print(f"  Created: {len(results)} sample files in samples/")

    def _prepare_sample_tasks(self, sample_headers):
        """
        Prepares tasks for parallel sample processing.
        Detects stereo pairs and generates unique filenames.
        """
        tasks = []
        processed = set()
        filename_counts = {}

        for idx, header in enumerate(sample_headers):
            if idx in processed or not any(header.values()):
                continue

            sample_link, sample_type = header["sample_link"], header["sample_type"]
            # sample_type: 1=mono, 2=right, 4=left, 8=linked, 0x8000+N=ROM
            is_stereo = sample_type in [2, 4] and sample_link < len(sample_headers)

            if is_stereo:
                # Prepare stereo task
                linked_idx = header["sample_link"]
                linked_header = sample_headers[linked_idx]
                left_h, right_h = (header, linked_header) if header["sample_type"] == 4 else (linked_header, header)

                # Determine base filename
                common_name = self._extract_common_name(left_h["name"], right_h["name"]) or f"sample_{idx}_{linked_idx}"
                base_filename = sanitize_filename(common_name)

                # Make filename unique
                if base_filename in filename_counts:
                    count = filename_counts[base_filename]
                    filename_counts[base_filename] += 1
                    final_filename = f"{base_filename}_{count}"
                    original_name = base_filename
                else:
                    filename_counts[base_filename] = 1
                    final_filename = base_filename
                    original_name = None

                tasks.append({
                    "type": "stereo",
                    "idx": idx,
                    "header": header,
                    "linked_idx": linked_idx,
                    "linked_header": linked_header,
                    "filename": final_filename,
                    "original_name": original_name
                })
                processed.add(idx)
                processed.add(linked_idx)
            else:
                # Prepare mono task
                base_filename = sanitize_filename(header["name"])

                # Make filename unique
                if base_filename in filename_counts:
                    count = filename_counts[base_filename]
                    filename_counts[base_filename] += 1
                    final_filename = f"{base_filename}_{count}"
                    original_name = base_filename
                else:
                    filename_counts[base_filename] = 1
                    final_filename = base_filename
                    original_name = None

                tasks.append({
                    "type": "mono",
                    "idx": idx,
                    "header": header,
                    "filename": final_filename,
                    "original_name": original_name
                })
                processed.add(idx)

        return tasks

    def _process_sample_tasks_parallel(self, tasks, samples_dir, sample_data, sample_headers):
        """
        Processes sample tasks in parallel with progress display.
        """
        total_tasks = len(tasks)
        print(f"  Processing {total_tasks} samples...")

        # Show warnings for duplicates
        for task in tasks:
            if task.get("original_name"):
                print(f"    WARNING: Duplicate sample name found. Original: \"{task['original_name']}\"")
                print(f"      -> Renaming to: \"{task['filename']}\"")

        results = []
        with ThreadPoolExecutor() as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    self._process_sample_task,
                    task,
                    samples_dir,
                    sample_data,
                    sample_headers
                ): task for task in tasks
            }

            # Process completed tasks with progress
            for completed, future in enumerate(as_completed(future_to_task), 1):
                try:
                    result = future.result()
                    if result:
                        results.append(result)

                        # Show progress inline
                        progress = (completed / total_tasks) * 100
                        print(f"    Progress: {completed}/{total_tasks} ({progress:.1f}%)", end="\r")
                except Exception as e:
                    task = future_to_task[future]
                    print(f"\n  ERROR processing sample {task.get('idx', 'unknown')}: {e}")

            # Print newline after progress is complete
            print()

        return results

    def _process_sample_task(self, task, samples_dir, sample_data, sample_headers):
        """
        Processes a single sample task(mono or stereo).
        Returns a dict with filename and id_mapping.
        """
        if task["type"] == "stereo":
            return self._process_stereo_sample(
                task["idx"],
                task["header"],
                task["linked_idx"],
                task["linked_header"],
                task["filename"],
                samples_dir,
                sample_data
            )
        else:
            return self._process_mono_sample(
                task["idx"],
                task["header"],
                task["filename"],
                samples_dir,
                sample_data
            )

    def _process_stereo_sample(self, idx, header, linked_idx, linked_header, filename, samples_dir, sample_data):
        """
        Processes a stereo sample(thread - safe version).
        """
        # Determine left and right channels
        left_h, right_h, left_idx, right_idx = (
            (header, linked_header, idx, linked_idx)
            if header["sample_type"] == 4
            else (linked_header, header, linked_idx, idx)
        )

        # Left channel PCM data
        left_pcm = np.frombuffer(sample_data[left_h["start"] * 2:left_h["end"] * 2], dtype=np.int16)
        # Right channel PCM data
        right_pcm = np.frombuffer(sample_data[right_h["start"] * 2:right_h["end"] * 2], dtype=np.int16)
        # Align lengths (to the shorter one)
        min_len = min(len(left_pcm), len(right_pcm))
        # Create stereo array (samples, channels)
        stereo_pcm = np.column_stack((left_pcm[:min_len], right_pcm[:min_len]))

        # Use the provided unique filename
        final_filename = filename

        # Write stereo FLAC file
        sf.write(samples_dir / f"{final_filename}.flac", stereo_pcm, left_h["sample_rate"], subtype="PCM_16")
        # Save metadata to JSON (as relative positions)
        self._write_sample_metadata(samples_dir / f"{final_filename}.json", final_filename, "stereo", len(left_pcm), left_h)

        return {
            "filename": final_filename,
            "id_mapping": {left_idx: final_filename, right_idx: final_filename}
        }

    def _process_mono_sample(self, idx, header, filename, samples_dir, sample_data):
        """
        Processes a mono sample(thread - safe version).
        """
        pcm_array = np.frombuffer(sample_data[header["start"] * 2:header["end"] * 2], dtype=np.int16)

        # Use the provided unique filename
        final_filename = filename

        # Write mono FLAC file
        sf.write(samples_dir / f"{final_filename}.flac", pcm_array, header["sample_rate"], subtype="PCM_16")
        # Save metadata to JSON (as relative positions)
        self._write_sample_metadata(samples_dir / f"{final_filename}.json", final_filename, "mono", len(pcm_array), header)

        return {
            "filename": final_filename,
            "id_mapping": {idx: final_filename}
        }

    def _write_sample_metadata(self, path, name, type, length, header):
        """
        Writes sample metadata to a JSON file.
        """
        metadata = {
            "sample_name": name,
            "sample_type": type,
            "start": 0,
            "end": length,
            "start_loop": header["start_loop"] - header["start"],
            "end_loop": header["end_loop"] - header["start"],
            "original_key": header["original_key"],
            "correction": header["correction"]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _export_instruments(self):
        """
        Exports JSON files to the instruments directory.
        """
        instruments_dir = self.output_dir / "instruments"
        instruments_dir.mkdir(exist_ok=True)
        inst_headers = self.parser.get_instrument_headers()
        inst_bags = self.parser.get_instrument_bags()
        inst_gens = self.parser.get_instrument_generators()
        inst_mods = self.parser.get_instrument_modulators()
        sample_headers = self.parser.get_sample_headers()
        for idx, inst in enumerate(inst_headers):
            if not any(inst.values()):
                continue
            # Get zone range for this instrument
            zones = self._get_zones(
                idx,
                inst_headers,
                inst_bags,
                inst_gens,
                inst_mods,
                lambda gens: self._parse_instrument_generators(gens, sample_headers)
            )
            inst_data = {
                "name": inst["name"],
                "zones": zones
            }
            # Filename (remove index)
            filename = f"{sanitize_filename(inst["name"])}.json"
            with open(instruments_dir / filename, "w", encoding="utf-8") as f:
                json.dump(inst_data, f, indent=2, ensure_ascii=False)
        print(f"  Created: {len(inst_headers)} instrument files in instruments/")

    def _export_presets(self):
        """
        Exports JSON files to the presets directory.
        """
        presets_dir = self.output_dir / "presets"
        presets_dir.mkdir(exist_ok=True)
        preset_headers = self.parser.get_preset_headers()
        preset_bags = self.parser.get_preset_bags()
        preset_gens = self.parser.get_preset_generators()
        preset_mods = self.parser.get_preset_modulators()
        inst_headers = self.parser.get_instrument_headers()
        for idx, preset in enumerate(preset_headers):
            if not any(preset.values()):
                continue
            # Get zone range for this preset
            zones = self._get_zones(
                idx,
                preset_headers,
                preset_bags,
                preset_gens,
                preset_mods,
                lambda gens: self._parse_preset_generators(gens, inst_headers)
            )
            preset_data = {
                "name": preset["name"],
                "bank": preset["bank"],
                "preset_number": preset["preset"],
                "library": preset["library"],
                "genre": preset["genre"],
                "morphology": preset["morphology"],
                "zones": zones
            }
            # Filename (keep bank-preset number, remove index)
            filename = f"{preset["bank"]:03d}-{preset["preset"]:03d}_{sanitize_filename(preset["name"])}.json"
            with open(presets_dir / filename, "w", encoding="utf-8") as f:
                json.dump(preset_data, f, indent=2, ensure_ascii=False)
        print(f"  Created: {len(preset_headers)} preset files in presets/")

    def _get_zones(self, header_idx, headers, bags, gens, mods, gen_parser_func):
        """
        Gets zones for an instrument or preset.
        """
        bag_start = headers[header_idx]["bag_ndx"]
        bag_end = headers[header_idx + 1]["bag_ndx"] if header_idx + 1 < len(headers) else len(bags) - 1
        zones = []
        for bag_idx in range(bag_start, bag_end):
            if bag_idx >= len(bags):
                break
            bag = bags[bag_idx]
            # Get generator and modulator ranges
            gen_start, mod_start = bag["gen_ndx"], bag["mod_ndx"]
            gen_end = bags[bag_idx + 1]["gen_ndx"] if bag_idx + 1 < len(bags) else len(gens)
            mod_end = bags[bag_idx + 1]["mod_ndx"] if bag_idx + 1 < len(bags) else len(mods)

            # Parse generators
            generators = gen_parser_func(gens[gen_start:gen_end])
            # Parse modulators
            modulators = mods[mod_start:mod_end]
            zone = {"generators": generators}
            if modulators:
                zone["modulators"] = modulators
            zones.append(zone)
        return zones

    def _parse_instrument_generators(self, gen_records, sample_headers):
        """
        Parses instrument generators.
        """
        generators = {}
        for gen in gen_records:
            gen_name = GENERATOR_NAMES.get(gen["oper"], f"unknown_{gen["oper"]}")
            amount = gen["amount"]
            # Special handling
            if gen["oper"] == 53:  # sampleID -> renamed to sample
                if amount < len(sample_headers):
                    sample_h = sample_headers[amount]
                    # Get from sample ID -> filename mapping (handles duplicates)
                    final_name = self.sample_id_to_filename.get(amount, sanitize_filename(sample_h["name"]))
                    generators["sample"] = final_name
                    # Save channel info for stereo
                    if sample_h["sample_type"] in [2, 4]:  # Stereo
                        generators["sample_channel"] = "right" if sample_h["sample_type"] == 2 else "left"
                else:
                    generators["sample"] = amount
            elif gen["oper"] in [43, 44]:  # keyRange, velRange
                lo = amount & 0xFF
                hi = (amount >> 8) & 0xFF
                generators[gen_name] = f"{lo}-{hi}"
            else:
                generators[gen_name] = amount
        return generators

    def _parse_preset_generators(self, gen_records, inst_headers):
        """
        Parses preset generators.
        """
        generators = {}
        for gen in gen_records:
            gen_name = GENERATOR_NAMES.get(gen["oper"], f"unknown_{gen["oper"]}")
            amount = gen["amount"]
            # Special handling
            if gen["oper"] == 41:  # instrument
                # Use instrument name
                generators[gen_name] = inst_headers[amount]["name"] if amount < len(inst_headers) else amount
            elif gen["oper"] in [43, 44]:  # keyRange, velRange
                lo = amount & 0xFF
                hi = (amount >> 8) & 0xFF
                generators[gen_name] = f"{lo}-{hi}"
            else:
                generators[gen_name] = amount
        return generators

    def _extract_common_name(self, left_name, right_name):
        """
        Extracts a common name from left and right sample names.
        """
        # Find common part (longest common prefix)
        common = ""
        for i, (lc, rc) in enumerate(zip(left_name, right_name)):
            if lc == rc:
                common += lc
            else:
                break
        # Remove trailing symbols
        return common.rstrip("_-( ") or left_name


def main():
    """
    Main function to run the SF2 decompiler.
    """
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <input.sf2> <output_directory>")
        sys.exit(1)

    sf2_file = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(sf2_file):
        print(f"Error: File not found - {sf2_file}")
        sys.exit(1)

    decompiler = SF2Decompiler(sf2_file, output_dir)
    decompiler.decompile()


if __name__ == "__main__":
    main()
