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
import struct
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


def read_chunk_header(f):
    """
    Reads a RIFF chunk header (ID and size) from a file.

    Args:
        f: The file object.

    Returns:
        A tuple containing the chunk ID and chunk size.
    """
    chunk_id = f.read(4)
    if len(chunk_id) < 4:
        raise EOFError("Unexpected end of file while reading chunk ID.")

    chunk_size_bytes = f.read(4)
    if len(chunk_size_bytes) < 4:
        raise EOFError("Unexpected end of file while reading chunk size.")

    chunk_size = struct.unpack("<I", chunk_size_bytes)[0]
    return chunk_id, chunk_size


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


class SF2Parser:
    """
    A parser for SF2 files.
    """

    def __init__(self, filepath):
        """
        Initializes the SF2Parser.

        Args:
            filepath: The path to the SF2 file.
        """
        self.filepath = filepath
        self.file = None
        self.info_data = {}
        self.sample_data = b""
        self.sample_data_24 = b""
        self.pdta = {}

    def parse(self):
        """
        Parses the entire SF2 file.
        """
        with open(self.filepath, "rb") as f:
            self.file = f

            # RIFF header
            riff_id = f.read(4)
            if riff_id != b"RIFF":
                raise ValueError("Not a RIFF file")

            file_size = struct.unpack("<I", f.read(4))[0]

            form_type = f.read(4)
            if form_type != b"sfbk":
                raise ValueError("Not a SoundFont file")

            # Parse the three main chunks
            self._parse_chunks()

    def _parse_chunks(self):
        """
        Parses INFO, sdta, and pdta chunks.
        """
        while True:
            try:
                chunk_id, chunk_size = read_chunk_header(self.file)
                if chunk_id == b"LIST":
                    list_type = self.file.read(4)
                    if list_type == b"INFO":
                        self._parse_info_list(chunk_size - 4)
                    elif list_type == b"sdta":
                        self._parse_sdta_list(chunk_size - 4)
                    elif list_type == b"pdta":
                        self._parse_pdta_list(chunk_size - 4)
                    else:
                        self.file.seek(chunk_size - 4, 1)  # Skip unknown list
                else:
                    self.file.seek(chunk_size, 1)  # Skip unknown chunk

                # Align to next word
                if chunk_size % 2:
                    self.file.seek(1, 1)

            except EOFError:
                break

    def _parse_info_list(self, size):
        """
        Parses the INFO-list chunk.
        """
        chunk_end = self.file.tell() + size
        while self.file.tell() < chunk_end:
            sub_id, sub_size = read_chunk_header(self.file)
            data = self.file.read(sub_size)

            # Handle padding
            if sub_size % 2:
                self.file.read(1)

            self._store_info_data(sub_id, data)

    def _store_info_data(self, sub_id, data):
        """
        Stores INFO data.
        """
        if sub_id == b"ifil":
            # Version info
            major, minor = struct.unpack("<HH", data)
            self.info_data["version"] = f"{major}.{minor:02d}"
        else:
            key = sub_id.decode("ascii", errors="ignore").rstrip("\x00")
            value = data.decode("ascii", errors="ignore").rstrip("\x00")
            if key == "isng":
                self.info_data["sound_engine"] = value
            elif key == "INAM":
                self.info_data["bank_name"] = value
            elif key == "ICOP":
                self.info_data["copyright"] = value
            elif key == "ICMT":
                self.info_data["comment"] = value
            elif key == "ISFT":
                self.info_data["software"] = value
            elif key == "ICRD":
                self.info_data["creation_date"] = value
            elif key == "IENG":
                self.info_data["engineer"] = value
            elif key == "IPRD":
                self.info_data["product"] = value
            else:
                # Other INFO chunks
                self.info_data[key.lower()] = value

    def _parse_sdta_list(self, size):
        """
        Parses the sdta-list chunk.
        """
        chunk_end = self.file.tell() + size
        while self.file.tell() < chunk_end:
            sub_id, sub_size = read_chunk_header(self.file)
            if sub_id == b"smpl":
                # 16-bit sample data
                self.sample_data = self.file.read(sub_size)
            elif sub_id == b"sm24":
                # 24-bit LSB (can be unimplemented)
                self.sample_data_24 = self.file.read(sub_size)
            else:
                # Skip unknown chunks
                self.file.seek(sub_size, 1)

            # Handle padding
            if sub_size % 2:
                self.file.read(1)

    def _parse_pdta_list(self, size):
        """
        Parses the pdta-list (Hydra) chunk.
        """
        chunk_end = self.file.tell() + size
        # Read the 9 sub-chunks in order
        hydra_chunks = {"phdr", "pbag", "pmod", "pgen", "inst", "ibag", "imod", "igen", "shdr"}
        while self.file.tell() < chunk_end:
            sub_id, sub_size = read_chunk_header(self.file)
            sub_id_str = sub_id.decode("ascii", errors="ignore")
            if sub_id_str in hydra_chunks:
                # Store data
                self.pdta[sub_id_str] = self.file.read(sub_size)
            else:
                # Skip unknown chunks
                self.file.seek(sub_size, 1)

            # Handle padding
            if sub_size % 2:
                self.file.read(1)

    def _get_pdta_records(self, chunk_name, record_size, terminator):
        """
        Gets records from a pdta sub-chunk.
        """
        data = self.pdta.get(chunk_name, b"")
        records = []
        for i in range(0, len(data), record_size):
            if i + record_size > len(data):
                break
            chunk = data[i:i + record_size]
            name_bytes = chunk[0:20]
            # Skip terminator record
            if name_bytes.startswith(terminator):
                break
            records.append(chunk)
        return records

    def get_preset_headers(self):
        """
        Gets preset headers.
        """
        # sfPresetHeader = 38 bytes
        records = self._get_pdta_records("phdr", 38, b"EOP")
        headers = []
        for r in records:
            name = r[0:20].decode("ascii", errors="ignore").rstrip("\x00")
            values = struct.unpack("<HHHIII", r[20:38])
            headers.append({
                "name": name,
                "preset": values[0],
                "bank": values[1],
                "bag_ndx": values[2],
                "library": values[3],
                "genre": values[4],
                "morphology": values[5]
            })
        return headers

    def get_instrument_headers(self):
        """
        Gets instrument headers.
        """
        # sfInst = 22 bytes
        records = self._get_pdta_records("inst", 22, b"EOI")
        headers = []
        for r in records:
            name = r[0:20].decode("ascii", errors="ignore").rstrip("\x00")
            bag_ndx = struct.unpack("<H", r[20:22])[0]
            headers.append({"name": name, "bag_ndx": bag_ndx})
        return headers

    def get_sample_headers(self):
        """
        Gets sample headers.
        """
        # sfSample = 46 bytes
        records = self._get_pdta_records("shdr", 46, b"EOS")
        headers = []
        for r in records:
            name = r[0:20].decode("ascii", errors="ignore").rstrip("\x00")
            values = struct.unpack("<IIIIIBbHH", r[20:46])
            headers.append({
                "name": name,
                "start": values[0],
                "end": values[1],
                "start_loop": values[2],
                "end_loop": values[3],
                "sample_rate": values[4],
                "original_key": values[5],
                "correction": values[6],
                "sample_link": values[7],
                "sample_type": values[8]
            })
        return headers

    def _get_bags(self, chunk_name):
        """
        Gets bags (zones).
        """
        data = self.pdta.get(chunk_name, b"")
        bags = []
        # sfPresetBag / sfInstBag = 4 bytes
        for i in range(0, len(data), 4):
            if i + 4 > len(data):
                break
            gen_ndx, mod_ndx = struct.unpack("<HH", data[i:i + 4])
            bags.append({"gen_ndx": gen_ndx, "mod_ndx": mod_ndx})
        return bags

    def get_preset_bags(self):
        return self._get_bags("pbag")

    def get_instrument_bags(self):
        return self._get_bags("ibag")

    def _get_generators(self, chunk_name):
        """
        Gets generators.
        """
        data = self.pdta.get(chunk_name, b"")
        generators = []
        # sfGenList = 4 bytes
        for i in range(0, len(data), 4):
            if i + 4 > len(data):
                break
            oper, amount = struct.unpack("<Hh", data[i:i + 4])
            generators.append({"oper": oper, "amount": amount})
        return generators

    def get_preset_generators(self):
        return self._get_generators("pgen")

    def get_instrument_generators(self):
        return self._get_generators("igen")

    def _get_modulators(self, chunk_name):
        """
        Gets modulators.
        """
        data = self.pdta.get(chunk_name, b"")
        modulators = []
        # sfModList = 10 bytes
        for i in range(0, len(data), 10):
            if i + 10 > len(data):
                break
            values = struct.unpack("<HHhHH", data[i:i + 10])
            modulators.append({
                "src_oper": values[0],
                "dest_oper": values[1],
                "amount": values[2],
                "amt_src_oper": values[3],
                "trans_oper": values[4]
            })
        return modulators

    def get_preset_modulators(self):
        return self._get_modulators("pmod")

    def get_instrument_modulators(self):
        return self._get_modulators("imod")


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
        Processes a single sample task (mono or stereo).
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
        Processes a stereo sample (thread-safe version).
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
        Processes a mono sample (thread-safe version).
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
