# Copyright (c) 2025 bakajikara
#
# This file is licensed under the MIT License (MIT).
# See the LICENSE file in the project root for the full license text.

"""
SoundFont Decompiler - Decompiles a SoundFont file into a directory structure.

This tool expands a SoundFont file into the following structure:
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
from abc import ABC, abstractmethod

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

from .parser import SoundFontParser
from .constants import GENERATOR_NAMES


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


class SoundFontDecompiler(ABC):
    """
    Decompiles a SoundFont file into a directory structure.
    Automatically handles SF2 (PCM) and SF3 (Ogg Vorbis) formats.
    """
    def __new__(cls, sf_path, output_dir, split_stereo=False):
        """
        Factory method to create a SoundFontDecompiler instance.
        """
        parser_for_check = SoundFontParser(sf_path)
        version = parser_for_check.get_version()

        if version.startswith("3"):
            instance = super().__new__(_SF3Decompiler)
        else:
            instance = super().__new__(_SF2Decompiler)

        return instance

    def __init__(self, sf_path, output_dir, split_stereo=False):
        """
        Initializes the SoundFont Decompiler.

        Args:
            sf_path: The path to the SoundFont file.
            output_dir: The output directory path.
        """
        self.sf_path = sf_path
        self.output_dir = Path(output_dir)
        self.parser = SoundFontParser(sf_path)
        self.sample_id_to_name = {}
        self.split_stereo = split_stereo

        # Track existing files in the output directory
        self.existing_files_map = {}  # Maps basename -> full path
        self.written_files = set()  # Track files written during this run

    def decompile(self):
        """
        Decompiles the file.
        """
        print(f"Parsing file: {self.sf_path}")
        self.parser.parse()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Decompiling to: {self.output_dir}")

        # Scan existing files in subdirectories
        self._scan_existing_files()

        # Decompile each part
        self._export_bank_info()
        self._export_samples()
        self._export_instruments()
        self._export_presets()

        # Clean up unwritten files
        self._cleanup_unwritten_files()

        print("Decompilation complete!")

    def _scan_existing_files(self):
        """
        Scans the output directory for existing files in samples/, instruments/, and presets/ subdirectories.
        Creates a mapping of (subdir_name, basename) -> full path for later lookup.
        Raises an error if duplicate filenames with the same extension are found within the same subdirectory hierarchy.
        """
        all_duplicates = []  # Collect all duplicates before raising error

        for subdir_name in ["samples", "instruments", "presets"]:
            subdir = self.output_dir / subdir_name
            if subdir.exists() and subdir.is_dir():
                # Track filenames within this subdirectory to detect duplicates
                seen_files = {}  # Maps (filename) -> path for duplicate detection

                for file_path in subdir.rglob("*"):
                    if file_path.is_file():
                        # Check for duplicate full filename (same name + same extension)
                        if file_path.name in seen_files:
                            all_duplicates.append((
                                subdir_name,
                                file_path.name,
                                seen_files[file_path.name],
                                file_path
                            ))
                        else:
                            seen_files[file_path.name] = file_path

                        # Store with full filename including extension
                        key_full = (subdir_name, file_path.name)
                        self.existing_files_map[key_full] = file_path

                        # Also store with stem only (without extension) for matching
                        key_stem = (subdir_name, file_path.stem)
                        # Only store if not already present to avoid overwriting
                        if key_stem not in self.existing_files_map:
                            self.existing_files_map[key_stem] = file_path

        # If duplicates were found, report all of them at once
        if all_duplicates:
            error_msg = "Duplicate filenames found in the output directory:\n\n"
            for subdir_name, filename, path1, path2 in all_duplicates:
                error_msg += f"Duplicate in {subdir_name}/: \"{filename}\"\n"
                error_msg += f"  1. {path1.relative_to(self.output_dir)}\n"
                error_msg += f"  2. {path2.relative_to(self.output_dir)}\n\n"
            error_msg += "Please reorganize your files to avoid duplicate names within the same subdirectory."
            raise ValueError(error_msg)

    def _find_existing_file(self, subdir_name, basename, extension):
        """
        Finds an existing file in the subdirectory hierarchy.

        Args:
            subdir_name: The subdirectory name (samples, instruments, presets)
            basename: The base filename without extension
            extension: The file extension (e.g., ".flac", ".json")

        Returns:
            Path object if found, None otherwise
        """
        full_filename = f"{basename}{extension}"

        # Try exact match with extension first (subdir_name + full filename)
        key_full = (subdir_name, full_filename)
        if key_full in self.existing_files_map:
            return self.existing_files_map[key_full]

        # Try basename match (for files with same stem but different extension)
        key_stem = (subdir_name, basename)
        if key_stem in self.existing_files_map:
            existing_path = self.existing_files_map[key_stem]
            # Verify it's in the correct subdirectory
            if subdir_name in existing_path.parts:
                return existing_path

        return None

    def _get_output_path(self, subdir_name, basename, extension):
        """
        Gets the output path for a file, either overwriting an existing file or creating at default location.

        Args:
            subdir_name: The subdirectory name (samples, instruments, presets)
            basename: The base filename without extension
            extension: The file extension (e.g., ".flac", ".json")

        Returns:
            Path object for the output file
        """
        existing = self._find_existing_file(subdir_name, basename, extension)

        if existing:
            # Remove extension and re-add to ensure correct extension
            output_path = existing.parent / f"{basename}{extension}"
            return output_path
        else:
            # Create at default location
            return self.output_dir / subdir_name / f"{basename}{extension}"

    def _cleanup_unwritten_files(self):
        """
        Removes files in samples/, instruments/, and presets/ that were not written during this decompilation.
        Only removes files with extensions recognized by the compiler.
        Does not remove files outside these directories or files with other extensions (e.g., README).
        """
        # Extensions recognized by the compiler
        recognized_extensions = {".json", ".flac", ".wav", ".ogg", ".oga", ".mp3"}  # Keep in sync with compiler
        removed_count = 0

        for subdir_name in ["samples", "instruments", "presets"]:
            subdir = self.output_dir / subdir_name
            if subdir.exists() and subdir.is_dir():
                for file_path in list(subdir.rglob("*")):
                    if file_path.is_file() and file_path not in self.written_files:
                        # Only remove files with recognized extensions
                        if file_path.suffix.lower() in recognized_extensions:
                            try:
                                file_path.unlink()
                                removed_count += 1
                                print(f"  Removed: {file_path.relative_to(self.output_dir)}")
                            except Exception as e:
                                print(f"  Warning: Could not remove {file_path}: {e}")

                # Remove empty directories
                for dir_path in sorted(subdir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                    if dir_path.is_dir() and not any(dir_path.iterdir()):
                        try:
                            dir_path.rmdir()
                        except Exception:
                            pass

        if removed_count > 0:
            print(f"  Cleaned up {removed_count} old file(s)")

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
        smpl_data = self.parser.smpl_data
        sm24_data = self.parser.sm24_data

        # Prepare tasks for parallel processing
        tasks = self._prepare_sample_tasks(sample_headers)

        # Process tasks in parallel
        results = self._process_sample_tasks_parallel(tasks, smpl_data, sm24_data, sample_headers)

        # Update sample ID to name mappings
        for result in results:
            self.sample_id_to_name.update(result)

        print(f"  Created: {len(results)} sample files in samples/")

    def _prepare_sample_tasks(self, sample_headers):
        """
        Prepares tasks for parallel sample processing.
        Detects stereo pairs and generates unique filenames.
        """
        tasks = []
        processed = set()
        basename_counts = {}

        for idx, header in enumerate(sample_headers):
            if idx in processed or not any(header.values()):
                continue

            # sample_type: 1=mono, 2=right, 4=left, 8=linked, 0x8000+N=ROM
            is_stereo = header["sample_type"] & 6
            is_pair_valid = False

            if is_stereo:
                linked_idx = header["sample_link"]
                if linked_idx < len(sample_headers) and linked_idx not in processed:
                    linked_header = sample_headers[linked_idx]

                    is_mutual_link = (linked_header["sample_link"] == idx)
                    is_correct_types = ((header["sample_type"] & 6) + (linked_header["sample_type"] & 6) == 6)

                    if is_mutual_link and is_correct_types:
                        is_pair_valid = True

            if is_pair_valid:
                # Prepare combined stereo task
                linked_idx = header["sample_link"]
                linked_header = sample_headers[linked_idx]
                left_h, right_h = (header, linked_header) if header["sample_type"] & 4 else (linked_header, header)

                # Determine basename
                common_name = self._extract_common_name(left_h["name"], right_h["name"]) or f"sample_{idx}_{linked_idx}"
                initial_basename = sanitize_filename(common_name)

                # Make basename unique
                if initial_basename in basename_counts:
                    count = basename_counts[initial_basename]
                    basename_counts[initial_basename] += 1
                    final_basename = f"{initial_basename}_{count}"
                    original_name = initial_basename
                else:
                    basename_counts[initial_basename] = 1
                    final_basename = initial_basename
                    original_name = None

                tasks.append({
                    "type": "combined_stereo",
                    "idx": idx,
                    "header": header,
                    "linked_idx": linked_idx,
                    "linked_header": linked_header,
                    "basename": final_basename,
                    "original_name": original_name
                })
                processed.add(idx)
                processed.add(linked_idx)
            else:
                if is_stereo:
                    print(f"    WARNING: Invalid or broken stereo pair found for sample \"{header["name"]}\". Treating as a single channel.")

                # Prepare mono task
                initial_basename = sanitize_filename(header["name"])

                # Make basename unique
                if initial_basename in basename_counts:
                    count = basename_counts[initial_basename]
                    basename_counts[initial_basename] += 1
                    final_basename = f"{initial_basename}_{count}"
                    original_name = initial_basename
                else:
                    basename_counts[initial_basename] = 1
                    final_basename = initial_basename
                    original_name = None

                tasks.append({
                    "type": "single_channel",
                    "idx": idx,
                    "header": header,
                    "basename": final_basename,
                    "original_name": original_name,
                    "is_broken_pair": bool(is_stereo)
                })
                processed.add(idx)

        return tasks

    def _process_sample_tasks_parallel(self, tasks, smpl_data, sm24_data, sample_headers):
        """
        Processes sample tasks in parallel with progress display.
        """
        total_tasks = len(tasks)
        print(f"  Processing {total_tasks} samples...")

        # Show warnings for duplicates
        for task in tasks:
            if task.get("original_name"):
                print(f"    WARNING: Duplicate sample name found. Original: \"{task["original_name"]}\"")
                print(f"      -> Renaming to: \"{task["basename"]}\"")

        results = []
        with ThreadPoolExecutor() as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    self._process_sample_task,
                    task,
                    smpl_data,
                    sm24_data,
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
                    print(f"\n  ERROR processing sample {task.get("idx", "unknown")}: {e}")

            # Print newline after progress is complete
            print()

        return results

    def _process_sample_task(self, task, smpl_data, sm24_data, sample_headers):
        """
        Processes a sample task (single or combined).
        Returns a mapping of sample IDs to basenames.
        """

        if task["type"] == "single_channel":
            return self._process_single_channel_task(task, smpl_data, sm24_data)
        elif task["type"] == "combined_stereo":
            return self._process_combined_stereo_task(task, smpl_data, sm24_data)

    def _process_single_channel_task(self, task, smpl_data, sm24_data):
        """
        Processes a single channel sample task.
        """
        return self._write_single_channel_sample(task["idx"], task["header"], task["basename"], smpl_data, sm24_data, task["is_broken_pair"])

    @abstractmethod
    def _process_combined_stereo_task(self, task, smpl_data, sm24_data):
        """
        Processes a combined stereo sample task.
        """
        pass

    def _get_left_right_pair(self, task):
        """
        Returns the left and right headers and indices for a stereo pair task.

        Returns:
            tuple: (left_header, right_header, left_idx, right_idx)
        """
        header = task["header"]
        linked_header = task["linked_header"]
        idx = task["idx"]
        linked_idx = task["linked_idx"]

        if header["sample_type"] & 4:
            # Left channel is primary
            return header, linked_header, idx, linked_idx
        else:
            # Right channel is primary
            return linked_header, header, linked_idx, idx

    def _write_single_channel_sample(self, idx, header, basename, smpl_data, sm24_data, is_broken_pair=False):
        """
        Processes a mono sample, or a single channel from a stereo pair.
        """
        output_name = basename
        sample_type = "mono"
        # Check if it's part of a stereo pair
        if header["sample_type"] & 4:
            sample_type = "stereo_left"
            if not is_broken_pair:
                output_name = f"{basename}_L"
        elif header["sample_type"] & 2:
            sample_type = "stereo_right"
            if not is_broken_pair:
                output_name = f"{basename}_R"

        # Get appropriate extension for this format
        extension = self._get_audio_extension()

        # Get output path (may be existing file location)
        audio_path = self._get_output_path("samples", output_name, extension)
        json_path = self._get_output_path("samples", output_name, ".json")

        # Write audio data
        self._write_single_channel_audio_data(audio_path, header, smpl_data, sm24_data)
        self.written_files.add(audio_path)

        # Save metadata to JSON (as relative positions)
        self._write_sample_metadata(json_path, basename, sample_type, header)
        self.written_files.add(json_path)

        return {idx: basename}

    @abstractmethod
    def _get_audio_extension(self):
        """
        Returns the audio file extension for this format.
        """
        pass

    @abstractmethod
    def _write_single_channel_audio_data(self, output_path, header, smpl_data, sm24_data):
        """
        Writes the audio data for a single channel to a specific path.
        """
        pass

    @abstractmethod
    def _write_sample_metadata(self, path, sample_name, sample_type, header):
        """
        Writes sample metadata to a JSON file.
        """
        pass

    def _export_instruments(self):
        """
        Exports JSON files to the instruments directory.
        """
        instruments_dir = self.output_dir / "instruments"
        instruments_dir.mkdir(exist_ok=True)
        inst_headers = self.parser.get_instrument_headers()
        sample_headers = self.parser.get_sample_headers()
        for idx, inst in enumerate(inst_headers):
            if not any(inst.values()):
                continue

            raw_zones = self.parser.get_instrument_zones(idx)
            translated_zones = []
            for zone in raw_zones:
                translated_zone = {
                    "generators": self._translate_instrument_generators(zone["generators"], sample_headers)
                }
                if zone["modulators"]:
                    translated_zone["modulators"] = zone["modulators"]
                translated_zones.append(translated_zone)

            inst_data = {
                "name": inst["name"],
                "zones": translated_zones
            }
            filename = sanitize_filename(inst["name"])
            output_path = self._get_output_path("instruments", filename, ".json")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(inst_data, f, indent=2, ensure_ascii=False)
            self.written_files.add(output_path)

        print(f"  Created: {len(inst_headers)} instrument files in instruments/")

    def _export_presets(self):
        """
        Exports JSON files to the presets directory.
        """
        presets_dir = self.output_dir / "presets"
        presets_dir.mkdir(exist_ok=True)
        preset_headers = self.parser.get_preset_headers()
        inst_headers = self.parser.get_instrument_headers()
        for idx, preset in enumerate(preset_headers):
            if not any(preset.values()):
                continue

            raw_zones = self.parser.get_preset_zones(idx)
            translated_zones = []
            for zone in raw_zones:
                translated_zone = {
                    "generators": self._translate_preset_generators(zone["generators"], inst_headers)
                }
                if zone["modulators"]:
                    translated_zone["modulators"] = zone["modulators"]
                translated_zones.append(translated_zone)

            preset_data = {
                "name": preset["name"],
                "bank": preset["bank"],
                "preset_number": preset["preset"],
                "library": preset["library"],
                "genre": preset["genre"],
                "morphology": preset["morphology"],
                "zones": translated_zones
            }
            filename = f"{preset["bank"]:03d}-{preset["preset"]:03d}_{sanitize_filename(preset["name"])}"
            output_path = self._get_output_path("presets", filename, ".json")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(preset_data, f, indent=2, ensure_ascii=False)
            self.written_files.add(output_path)

        print(f"  Created: {len(preset_headers)} preset files in presets/")

    def _translate_instrument_generators(self, gen_records, sample_headers):
        """
        Translate instrument generator records into a dict mapping generator names
        to meaningful values.

        Special handling:
        - sampleID (oper 53): maps sample index to the exported sample filename
          and records stereo channel info ("left"/"right") when applicable.
        - keyRange / velRange (oper 43 / 44): converts the 16-bit range value
          into a "lo-hi" string.
        - Unknown generator ops are returned using a "unknown_<oper>" name.
        """
        generators = {}
        for gen in gen_records:
            gen_name = GENERATOR_NAMES.get(gen["oper"], f"unknown_{gen["oper"]}")
            amount = gen["amount"]
            # Special handling
            if gen["oper"] == 53:  # sampleID -> renamed to sample
                if amount < len(sample_headers):
                    sample_h = sample_headers[amount]
                    # Get from sample ID -> name mapping (handles duplicates)
                    final_name = self.sample_id_to_name.get(amount, sanitize_filename(sample_h["name"]))
                    generators["sample"] = final_name
                    # Save channel info for stereo
                    if sample_h["sample_type"] & 6:  # Stereo
                        generators["sample_channel"] = "left" if sample_h["sample_type"] & 4 else "right"
                else:
                    generators["sample"] = amount
            elif gen["oper"] in [43, 44]:  # keyRange, velRange
                lo = amount & 0xFF
                hi = (amount >> 8) & 0xFF
                generators[gen_name] = f"{lo}-{hi}"
            else:
                generators[gen_name] = amount
        return generators

    def _translate_preset_generators(self, gen_records, inst_headers):
        """
        Translate preset generator records into a dict mapping generator names
        to meaningful values.

        Special handling:
        - instrument (oper 41): map instrument index to instrument name when available.
        - keyRange / velRange (oper 43 / 44): convert 16-bit range value into "lo-hi" string.
        - Unknown generator ops are returned using a "unknown_<oper>" key.
        """
        generators = {}
        for gen in gen_records:
            gen_name = GENERATOR_NAMES.get(gen["oper"], f"unknown_{gen["oper"]}")
            amount = gen["amount"]
            # Special handling
            if gen["oper"] == 41:  # instrument
                # Use instrument name if index is valid
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


class _SF2Decompiler(SoundFontDecompiler):
    """
    Decompiler for SF2 files (PCM audio).
    """

    def _get_audio_extension(self):
        """
        Returns the audio file extension for SF2 format.
        """
        return ".flac"

    def _process_combined_stereo_task(self, task, smpl_data, sm24_data):
        """
        Processes a combined stereo sample task.
        """
        basename = task["basename"]

        left_h, right_h, left_idx, right_idx = self._get_left_right_pair(task)

        if self.split_stereo:
            # Split stereo into separate files
            left_result = self._write_single_channel_sample(left_idx, left_h, basename, smpl_data, sm24_data)
            right_result = self._write_single_channel_sample(right_idx, right_h, basename, smpl_data, sm24_data)
            # Combine results
            combined_result = {}
            combined_result.update(left_result)
            combined_result.update(right_result)
            return combined_result
        else:
            # Combined stereo file
            return self._write_combined_stereo_sample(left_idx, left_h, right_idx, right_h, basename, smpl_data, sm24_data)

    def _write_single_channel_audio_data(self, output_path, header, smpl_data, sm24_data):
        """
        Writes the audio data for a single channel to a FLAC file.
        """
        audio_data = smpl_data[header["start"] * 2:header["end"] * 2]
        audio_data_24_lsb = sm24_data[header["start"]:header["end"]] if sm24_data else b""
        is_24bit = any(audio_data_24_lsb)

        audio_pcm = np.frombuffer(audio_data, dtype=np.int16)

        # Handle 24-bit LSB data if available
        if is_24bit:
            audio_pcm_24_lsb = np.frombuffer(audio_data_24_lsb, dtype=np.uint8)
            audio_pcm = (audio_pcm.astype(np.int32) << 16) + (audio_pcm_24_lsb.astype(np.int32) << 8)

        # Write mono FLAC file
        sf.write(output_path, audio_pcm, header["sample_rate"], subtype="PCM_24" if is_24bit else "PCM_16")

    def _write_combined_stereo_sample(self, left_idx, left_h, right_idx, right_h, basename, smpl_data, sm24_data):
        """
        Writes a combined stereo sample to a FLAC file.
        """
        # Get output paths
        audio_path = self._get_output_path("samples", basename, ".flac")
        json_path = self._get_output_path("samples", basename, ".json")

        # Write audio data
        self._write_combined_stereo_audio_data(audio_path, left_h, right_h, smpl_data, sm24_data)
        self.written_files.add(audio_path)

        # Save metadata to JSON (as relative positions)
        self._write_sample_metadata(json_path, basename, "stereo", left_h)
        self.written_files.add(json_path)

        return {
            left_idx: basename,
            right_idx: basename
        }

    def _write_combined_stereo_audio_data(self, output_path, left_header, right_header, smpl_data, sm24_data):
        # Extract left channel data
        left_audio_data = smpl_data[left_header["start"] * 2:left_header["end"] * 2]
        left_audio_data_24_lsb = sm24_data[left_header["start"]:left_header["end"]] if sm24_data else b""
        left_audio_pcm = np.frombuffer(left_audio_data, dtype=np.int16)

        # Extract right channel data
        right_audio_data = smpl_data[right_header["start"] * 2:right_header["end"] * 2]
        right_audio_data_24_lsb = sm24_data[right_header["start"]:right_header["end"]] if sm24_data else b""
        right_audio_pcm = np.frombuffer(right_audio_data, dtype=np.int16)

        # Handle 24-bit LSB data if available
        is_24bit = any(left_audio_data_24_lsb) and any(right_audio_data_24_lsb)
        if is_24bit:
            left_audio_pcm_24_lsb = np.frombuffer(left_audio_data_24_lsb, dtype=np.uint8)
            left_audio_pcm = (left_audio_pcm.astype(np.int32) << 16) + (left_audio_pcm_24_lsb.astype(np.int32) << 8)
            right_audio_pcm_24_lsb = np.frombuffer(right_audio_data_24_lsb, dtype=np.uint8)
            right_audio_pcm = (right_audio_pcm.astype(np.int32) << 16) + (right_audio_pcm_24_lsb.astype(np.int32) << 8)

        # Ensure both channels have the same length
        min_length = min(len(left_audio_pcm), len(right_audio_pcm))
        left_audio_pcm = left_audio_pcm[:min_length]
        right_audio_pcm = right_audio_pcm[:min_length]

        # Combine into stereo array
        stereo_pcm = np.column_stack((left_audio_pcm, right_audio_pcm))

        # Write stereo FLAC file
        sf.write(output_path, stereo_pcm, left_header["sample_rate"], subtype="PCM_24" if is_24bit else "PCM_16")

    def _write_sample_metadata(self, path, sample_name, sample_type, header):
        """
        Writes sample metadata to a JSON file.
        Note: start/end are omitted as they will be calculated during compilation.
        """
        metadata = {
            "sample_name": sample_name,
            "sample_type": sample_type,
            "start_loop": header["start_loop"] - header["start"],
            "end_loop": header["end_loop"] - header["start"],
            "original_key": header["original_key"],
            "correction": header["correction"]
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


class _SF3Decompiler(SoundFontDecompiler):
    """
    Decompiler for SF3 files (Ogg Vorbis audio).
    """

    def _get_audio_extension(self):
        """
        Returns the audio file extension for SF3 format.
        """
        return ".ogg"

    def _process_combined_stereo_task(self, task, smpl_data, sm24_data):
        """
        Processes a combined stereo sample task.
        Always splits stereo into separate files for SF3.
        """
        basename = task["basename"]

        left_h, right_h, left_idx, right_idx = self._get_left_right_pair(task)

        # Always split stereo into separate files
        left_result = self._write_single_channel_sample(left_idx, left_h, basename, smpl_data, sm24_data)
        right_result = self._write_single_channel_sample(right_idx, right_h, basename, smpl_data, sm24_data)
        # Combine results
        combined_result = {}
        combined_result.update(left_result)
        combined_result.update(right_result)
        return combined_result

    def _write_single_channel_audio_data(self, output_path, header, smpl_data, sm24_data):
        """
        Writes the audio data for a single channel to an Ogg Vorbis file.
        """
        ogg_data = smpl_data[header["start"]:header["end"]]

        # Write raw Ogg Vorbis bytes without re-encoding
        with open(output_path, "wb") as f:
            f.write(ogg_data)

    def _write_sample_metadata(self, path, sample_name, sample_type, header):
        """
        Writes sample metadata to a JSON file.
        For SF3, loop positions are stored as relative offsets.
        """
        metadata = {
            "sample_name": sample_name,
            "sample_type": sample_type,
            "start_loop": header["start_loop"],
            "end_loop": header["end_loop"],
            "original_key": header["original_key"],
            "correction": header["correction"]
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    """
    Main function to run the SoundFont decompiler.
    """
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <input_file> <output_directory>")
        sys.exit(1)

    sf_file = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(sf_file):
        print(f"Error: File not found - {sf_file}")
        sys.exit(1)

    try:
        decompiler = SoundFontDecompiler(sf_file, output_dir)
        decompiler.decompile()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
