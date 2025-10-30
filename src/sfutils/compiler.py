#!/usr/bin/env python3
"""
SF2 Compiler - Rebuilds a SoundFont2 file from an expanded directory structure.

This tool generates an SF2 file from the following structure:
- info.json: Metadata
- samples/: Audio files (FLAC, WAV, etc.)
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
    print("Error: soundfile library is required.")
    print("Install it with: pip install soundfile")
    sys.exit(1)

from .constants import GENERATOR_IDS


def make_chunk(chunk_id, data):
    """
    Creates a RIFF chunk with the given ID and data.
    Automatically adds padding if the data size is odd.

    Args:
        chunk_id: The 4-byte chunk ID.
        data: The chunk's data.

    Returns:
        The created chunk as a bytes object.
    """
    if len(chunk_id) != 4:
        raise ValueError("Chunk ID must be 4 bytes long.")

    size = len(data)
    packed_data = chunk_id + struct.pack("<I", size) + data

    # Add padding if size is odd
    if size % 2:
        packed_data += b"\x00"

    return packed_data


def make_list_chunk(list_type, data):
    """
    Creates a LIST chunk with the given list type and data.

    Args:
        list_type: The 4-byte list type ID (e.g., b"INFO", b"pdta").
        data: The internal data.

    Returns:
        The created LIST chunk as a bytes object.
    """
    if len(list_type) != 4:
        raise ValueError("List type ID must be 4 bytes long.")

    # Internal data starts with the list type ID
    list_data = list_type + data
    return make_chunk(b"LIST", list_data)


def make_zstr(text, encoding="ascii"):
    """
    Creates a zero-terminated string compliant with the RIFF specification.
    Adjusts the total byte count to be even.

    Args:
        text: The string.
        encoding: The encoding (default: "ascii").

    Returns:
        The zero-terminated bytes.
    """
    encoded = text.encode(encoding)

    # If string length is odd, add one terminator (total even)
    # If string length is even, add two terminators (total even)
    if len(encoded) % 2 == 1:
        return encoded + b"\x00"
    else:
        return encoded + b"\x00\x00"


class SF2Compiler:
    """
    A class to compile a directory structure into an SF2 file.
    """

    # Padding between samples in sample units (minimum 46 samples or 92 bytes)
    SAMPLE_PADDING = 46

    def __init__(self, input_dir, output_sf2):
        """
        Initializes the SF2Compiler.

        Args:
            input_dir: The input directory path.
            output_sf2: The output SF2 file path.
        """
        self.input_dir = Path(input_dir)
        self.output_sf2 = output_sf2

        # Data storage
        self.bank_info = {}
        self.samples = []
        self.instruments = []
        self.presets = []

    def compile(self):
        """
        Generates an SF2 file from the directory structure.
        """
        print(f"Compiling from: {self.input_dir}")

        # Load all parts
        self._load_bank_info()
        self._load_samples()
        self._fix_stereo_links()  # Fix stereo sample links after loading
        self._load_instruments()
        self._load_presets()

        # Generate the SF2 file
        print(f"Writing SF2 file: {self.output_sf2}")
        self._write_sf2_file()

        print("Compilation complete!")

    def _fix_stereo_links(self):
        """
        Fixes stereo sample links after parallel loading.
        Stereo pairs need to reference each other by index.
        """
        # Group stereo samples by audio path and name
        stereo_groups = {}
        for idx, sample in enumerate(self.samples):
            if sample.get("_is_stereo"):
                key = (sample["_audio_path"], sample["sample_name"])
                if key not in stereo_groups:
                    stereo_groups[key] = []
                stereo_groups[key].append(idx)

        # Set mutual links for each stereo pair
        for indices in stereo_groups.values():
            if len(indices) == 2:
                left_idx, right_idx = indices[0], indices[1]
                # Determine which is left and which is right
                if self.samples[left_idx]["_channel"] == "left":
                    self.samples[left_idx]["sample_link"] = right_idx
                    self.samples[right_idx]["sample_link"] = left_idx
                else:
                    self.samples[left_idx]["sample_link"] = right_idx
                    self.samples[right_idx]["sample_link"] = left_idx

    def _load_bank_info(self):
        """
        Loads info.json.
        """
        info_path = self.input_dir / "info.json"

        if not info_path.exists():
            raise FileNotFoundError(f"info.json not found in {self.input_dir}")

        with open(info_path, "r", encoding="utf-8") as f:
            self.bank_info = json.load(f)

        print(f"  Loaded: info.json")

    def _load_samples(self):
        """
        Loads sample metadata from the samples directory.
        PCM data is read later for performance.
        """
        samples_dir = self.input_dir / "samples"

        if not samples_dir.exists():
            raise FileNotFoundError(f"samples directory not found in {self.input_dir}")

        # Load JSON files
        json_files = sorted(samples_dir.glob("*.json"))  # sorted for consistent order

        total_files = len(json_files)
        print(f"  Loading {total_files} sample metadata files...")

        # Process in parallel with progress display, preserving order
        sample_entries = []
        with ThreadPoolExecutor() as executor:
            # Submit all tasks and map futures to their index
            future_to_index = {
                executor.submit(self._load_sample_metadata, json_path): idx
                for idx, json_path in enumerate(json_files)
            }

            # Collect results with their original indices
            results_with_index = []
            for completed, future in enumerate(as_completed(future_to_index), 1):
                try:
                    entries = future.result()
                    idx = future_to_index[future]
                    results_with_index.append((idx, entries))

                    # Show progress inline
                    progress = (completed / total_files) * 100
                    print(f"    Progress: {completed}/{total_files} ({progress:.1f}%)", end="\r")
                except Exception as e:
                    idx = future_to_index[future]
                    print(f"\n  ERROR loading {json_files[idx]}: {e}")

            print()  # Newline after progress

            # Sort by original index to preserve order
            results_with_index.sort(key=lambda x: x[0])
            for _, entries in results_with_index:
                sample_entries.extend(entries)

        # Add samples in the order they were loaded
        self.samples = sample_entries

        print(f"  Loaded: {len(self.samples)} sample entries from samples/")

    def _load_sample_metadata(self, json_path):
        """
        Loads metadata for a single sample (or stereo pair).
        Returns a list of sample entries (1 for mono, 2 for stereo).
        """
        # Load metadata from JSON
        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Find the corresponding audio file path (FLAC, WAV, etc.)
        audio_path = None
        for ext in [".flac", ".wav", ".ogg", ".aiff", ".aif"]:
            candidate = json_path.with_suffix(ext)
            if candidate.exists():
                audio_path = candidate
                break

        if audio_path is None:
            raise FileNotFoundError(f"Audio file not found for: {json_path}")

        sample_type = metadata.get("sample_type", "mono")

        if sample_type == "stereo":
            return self._create_stereo_sample_entries(metadata, audio_path)
        else:
            return [self._create_mono_sample_entry(metadata, audio_path)]

    def _create_stereo_sample_entries(self, metadata, audio_path):
        """
        Creates stereo sample entries (left and right).
        """
        # Stereo sample: split into two samples
        start = metadata.get("start", 0)
        end = metadata.get("end", 0)
        start_loop = metadata.get("start_loop", 0)
        end_loop = metadata.get("end_loop", 0)
        original_key = metadata.get("original_key", 60)
        correction = metadata.get("correction", 0)

        # Left channel sample
        left_sample = {
            "sample_name": metadata["sample_name"],
            "start": start,
            "end": end,
            "start_loop": start_loop,
            "end_loop": end_loop,
            "original_key": original_key,
            "correction": correction,
            "sample_link": None,  # Will be set later
            "sample_type": 4,
            "_audio_path": audio_path,
            "_channel": "left",
            "_is_stereo": True
        }

        # Right channel sample
        right_sample = {
            "sample_name": metadata["sample_name"],
            "start": start,
            "end": end,
            "start_loop": start_loop,
            "end_loop": end_loop,
            "original_key": original_key,
            "correction": correction,
            "sample_link": None,  # Will be set later
            "sample_type": 2,
            "_audio_path": audio_path,
            "_channel": "right",
            "_is_stereo": True
        }

        # Note: sample_link indices will be fixed after all samples are collected
        return [left_sample, right_sample]

    def _create_mono_sample_entry(self, metadata, audio_path):
        """
        Creates a mono sample entry.
        """
        return {
            "sample_name": metadata["sample_name"],
            "start": metadata.get("start", 0),
            "end": metadata.get("end", 0),
            "start_loop": metadata.get("start_loop", 0),
            "end_loop": metadata.get("end_loop", 0),
            "original_key": metadata.get("original_key", 60),
            "correction": metadata.get("correction", 0),
            "sample_link": 0,
            "sample_type": 1,
            "_audio_path": audio_path,
            "_channel": None,
            "_is_stereo": False
        }

    def _read_pcm_data(self, audio_path, channel=None):
        """
        Reads PCM data from an audio file (FLAC, WAV, etc.) when needed.

        Args:
            audio_path: The path to the audio file.
            channel: "left", "right", or None (for mono).
        """
        try:
            # Read with soundfile (supports FLAC, WAV, OGG, AIFF, etc.)
            data, samplerate = sf.read(audio_path, dtype="int16")

            # Separate channels
            if channel == "left":
                if len(data.shape) == 2:
                    data = data[:, 0]
            elif channel == "right":
                if len(data.shape) == 2:
                    data = data[:, 1]

            # Convert NumPy array to bytes
            return data.tobytes(), samplerate
        except Exception as e:
            raise ValueError(f"Failed to read audio file {audio_path}: {e}")

    def _load_instruments(self):
        """
        Loads JSON files from the instruments directory.
        """
        instruments_dir = self.input_dir / "instruments"

        if not instruments_dir.exists():
            raise FileNotFoundError(f"instruments directory not found in {self.input_dir}")

        json_files = sorted(instruments_dir.glob("*.json"))
        total_files = len(json_files)

        # Process in parallel with progress display, preserving order
        with ThreadPoolExecutor() as executor:
            # Submit all tasks and map futures to their index
            future_to_index = {
                executor.submit(self._load_json_file, json_path): idx
                for idx, json_path in enumerate(json_files)
            }

            # Collect results with their original indices
            results_with_index = []
            for completed, future in enumerate(as_completed(future_to_index), 1):
                try:
                    data = future.result()
                    idx = future_to_index[future]
                    results_with_index.append((idx, data))

                    # Show progress inline
                    progress = (completed / total_files) * 100
                except Exception as e:
                    idx = future_to_index[future]
                    print(f"  ERROR loading {json_files[idx]}: {e}")

            # Sort by original index to preserve order
            results_with_index.sort(key=lambda x: x[0])
            self.instruments = [data for _, data in results_with_index]

        print(f"  Loaded: {len(self.instruments)} instrument files from instruments/")

    def _load_presets(self):
        """
        Loads JSON files from the presets directory.
        """
        presets_dir = self.input_dir / "presets"

        if not presets_dir.exists():
            raise FileNotFoundError(f"presets directory not found in {self.input_dir}")

        json_files = sorted(presets_dir.glob("*.json"))
        total_files = len(json_files)

        # Process in parallel with progress display, preserving order
        with ThreadPoolExecutor() as executor:
            # Submit all tasks and map futures to their index
            future_to_index = {
                executor.submit(self._load_json_file, json_path): idx
                for idx, json_path in enumerate(json_files)
            }

            # Collect results with their original indices
            results_with_index = []
            for completed, future in enumerate(as_completed(future_to_index), 1):
                try:
                    data = future.result()
                    idx = future_to_index[future]
                    results_with_index.append((idx, data))

                    # Show progress inline
                    progress = (completed / total_files) * 100
                except Exception as e:
                    idx = future_to_index[future]
                    print(f"  ERROR loading {json_files[idx]}: {e}")

            # Sort by original index to preserve order
            results_with_index.sort(key=lambda x: x[0])
            self.presets = [data for _, data in results_with_index]

        print(f"  Loaded: {len(self.presets)} preset files from presets/")

    def _load_json_file(self, json_path):
        """
        Loads a single JSON file.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_sf2_file(self):
        """
        Writes the SF2 file.
        """
        with open(self.output_sf2, "wb") as f:
            # RIFF header (size updated later)
            f.write(b"RIFF")
            riff_size_pos = f.tell()
            f.write(struct.pack("<I", 0))
            f.write(b"sfbk")

            # INFO-list
            info_chunk = self._build_info_chunk()
            f.write(info_chunk)

            # sdta-list (written directly for memory efficiency)
            self._write_sdta_chunk_direct(f)

            # pdta-list
            pdta_chunk = self._build_pdta_chunk()
            f.write(pdta_chunk)

            # Update RIFF size
            file_size = f.tell()
            f.seek(riff_size_pos)
            f.write(struct.pack("<I", file_size - 8))

    def _build_info_chunk(self):
        """
        Builds the INFO-list chunk.
        """
        data = b""
        # Version info (required, first)
        version = self.bank_info.get("version", "2.04")
        major, minor = version.split(".")
        ifil_data = struct.pack("<HH", int(major), int(minor))
        data += make_chunk(b"ifil", ifil_data)

        # Sound engine (required, second)
        sound_engine = self.bank_info.get("sound_engine", "EMU8000")
        data += make_chunk(b"isng", make_zstr(sound_engine))

        # Bank name (required, third)
        bank_name = self.bank_info.get("bank_name", "Untitled")
        data += make_chunk(b"INAM", make_zstr(bank_name))

        # Output in SF2 spec order (same as original files)
        info_order = ["creation_date", "engineer", "product", "copyright", "comment", "software"]
        info_map = {
            "creation_date": b"ICRD",
            "engineer": b"IENG",
            "product": b"IPRD",
            "copyright": b"ICOP",
            "comment": b"ICMT",
            "software": b"ISFT",
        }

        for key in info_order:
            if key in self.bank_info:
                data += make_chunk(info_map[key], make_zstr(self.bank_info[key]))

        # Wrap as a LIST chunk
        return make_list_chunk(b"INFO", data)

    def _write_sdta_chunk_direct(self, f):
        """
        Writes the sdta-list chunk directly to the file for memory efficiency.
        """
        # LIST chunk header
        f.write(b"LIST")
        list_size_pos = f.tell()
        f.write(struct.pack("<I", 0))
        f.write(b"sdta")

        # smpl chunk header
        f.write(b"smpl")
        smpl_size_pos = f.tell()
        f.write(struct.pack("<I", 0))

        smpl_start = f.tell()

        # Write sample data sequentially with progress display
        padding = b"\x00" * self.SAMPLE_PADDING * 2
        current_offset = 0
        total_samples = len(self.samples)

        print(f"  Writing {total_samples} samples to SF2...")

        for idx, sample in enumerate(self.samples, 1):
            # Read PCM data only when needed, passing channel info
            channel = sample.get("_channel")
            pcm, sample_rate = self._read_pcm_data(sample["_audio_path"], channel)
            num_samples = len(pcm) // 2

            # Calculate absolute positions
            sample["_absolute_start"] = current_offset + sample["start"]
            sample["_absolute_end"] = current_offset + sample["end"]
            sample["_absolute_start_loop"] = current_offset + sample["start_loop"]
            sample["_absolute_end_loop"] = current_offset + sample["end_loop"]
            sample["_sample_rate"] = sample_rate

            # Write data
            f.write(pcm)
            f.write(padding)

            # Update offset for the next sample
            current_offset += num_samples + self.SAMPLE_PADDING

            # Show progress inline
            progress = (idx / total_samples) * 100
            print(f"    Progress: {idx}/{total_samples} ({progress:.1f}%)", end="\r")

        print()  # Newline after progress

        # Calculate and update sizes
        smpl_end = f.tell()
        smpl_size = smpl_end - smpl_start

        # Update smpl chunk size
        f.seek(smpl_size_pos)
        f.write(struct.pack("<I", smpl_size))

        # Add padding if size is odd
        f.seek(smpl_end)
        if smpl_size % 2:
            f.write(b"\x00")

        # Update LIST chunk size
        list_end = f.tell()
        list_size = list_end - list_size_pos - 4
        f.seek(list_size_pos)
        f.write(struct.pack("<I", list_size))

        # Return file pointer to the end
        f.seek(list_end)

    def _build_shdr_chunk(self):
        """
        Builds the shdr chunk.
        """
        shdr_data = []
        for idx, sample in enumerate(self.samples):
            # Dynamically generate sample name (add left/right suffixes)
            base_name = sample["sample_name"]
            if sample.get("_is_stereo"):
                channel = sample.get("_channel")
                name = f"{base_name}_L"[:20] if channel == "left" else f"{base_name}_R"[:20]
            else:
                name = base_name[:20]

            name_bytes = name.ljust(20, "\x00").encode("ascii")
            # Use absolute positions calculated in _write_sdta_chunk_direct
            start = sample.get("_absolute_start", 0)
            end = sample.get("_absolute_end", 0)
            start_loop = sample.get("_absolute_start_loop", 0)
            end_loop = sample.get("_absolute_end_loop", 0)
            sample_rate = sample.get("_sample_rate", 44100)

            shdr_record = struct.pack(
                "<20sIIIIIBBHH",
                name_bytes,
                start,
                end,
                start_loop,
                end_loop,
                sample_rate,
                sample["original_key"],
                sample["correction"] & 0xFF,
                sample["sample_link"],
                sample["sample_type"]
            )
            shdr_data.append(shdr_record)

        # Terminator ("EOS")
        shdr_data.append(b"EOS".ljust(20, b"\x00") + b"\x00" * 26)
        return b"".join(shdr_data)

    def _build_pdta_chunk(self):
        """
        Builds the pdta-list (Hydra) chunk.
        """
        # Build each part of Hydra
        phdr_data, pbag_data, pmod_data, pgen_data = self._build_presets_chunk()
        inst_data, ibag_data, imod_data, igen_data = self._build_instruments_chunk()
        shdr_chunk = self._build_shdr_chunk()

        # Build each chunk
        result = b""
        result += make_chunk(b"phdr", b"".join(phdr_data))
        result += make_chunk(b"pbag", b"".join(pbag_data))
        result += make_chunk(b"pmod", b"".join(pmod_data))
        result += make_chunk(b"pgen", b"".join(pgen_data))
        result += make_chunk(b"inst", b"".join(inst_data))
        result += make_chunk(b"ibag", b"".join(ibag_data))
        result += make_chunk(b"imod", b"".join(imod_data))
        result += make_chunk(b"igen", b"".join(igen_data))
        result += make_chunk(b"shdr", shdr_chunk)

        # Wrap as a LIST chunk
        return make_list_chunk(b"pdta", result)

    def _build_instruments_chunk(self):
        """
        Builds the instrument-related chunks.
        """
        inst_data, ibag_data, imod_data, igen_data = [], [], [], []
        # Sample name + channel to index mapping
        sample_name_to_id = {}
        for i, s in enumerate(self.samples):
            base_name = s["sample_name"]
            if not s.get("_is_stereo"):
                sample_name_to_id[base_name] = i
            channel = s.get("_channel")
            if channel:
                sample_name_to_id[(base_name, channel)] = i

        for inst in self.instruments:
            name = inst["name"][:20].ljust(20, "\x00").encode("ascii")
            bag_ndx = len(ibag_data)
            inst_record = struct.pack("<20sH", name, bag_ndx)
            inst_data.append(inst_record)

            # Process zones
            for zone in inst["zones"]:
                gen_ndx, mod_ndx = len(igen_data), len(imod_data)
                ibag_data.append(struct.pack("<HH", gen_ndx, mod_ndx))

                # Add modulators
                if "modulators" in zone:
                    for mod in zone["modulators"]:
                        imod_data.append(struct.pack(
                            "<HHhHH",
                            mod["src_oper"],
                            mod["dest_oper"],
                            mod["amount"],
                            mod["amt_src_oper"],
                            mod["trans_oper"]
                        ))

                # Add generators
                self._add_instrument_generators(igen_data, zone["generators"], sample_name_to_id)

        # Terminators
        inst_data.append(b"EOI".ljust(20, b"\x00") + struct.pack("<H", len(ibag_data)))
        ibag_data.append(struct.pack("<HH", len(igen_data), len(imod_data)))
        imod_data.append(struct.pack("<HHhHH", 0, 0, 0, 0, 0))
        igen_data.append(struct.pack("<Hh", 0, 0))

        return inst_data, ibag_data, imod_data, igen_data

    def _add_instrument_generators(self, igen_data, generators, sample_name_to_id):
        """
        Adds instrument generators.
        """
        # keyRange and velRange first
        if "keyRange" in generators:
            lo, hi = map(int, generators["keyRange"].split("-"))
            igen_data.append(struct.pack("<Hh", 43, lo | (hi << 8)))
        if "velRange" in generators:
            lo, hi = map(int, generators["velRange"].split("-"))
            igen_data.append(struct.pack("<Hh", 44, lo | (hi << 8)))

        # Other generators
        for gen_name, gen_value in generators.items():
            if gen_name in ["keyRange", "velRange", "sample"]:
                continue
            gen_id = GENERATOR_IDS.get(gen_name)
            if gen_id is not None:
                igen_data.append(struct.pack("<Hh", gen_id, gen_value))

        # sample (formerly sampleID) last
        if "sample" in generators:
            sample_name = generators["sample"]
            sample_channel = generators.get("sample_channel")
            key = (sample_name, sample_channel) if sample_channel else sample_name
            sample_id = sample_name_to_id.get(key, 0)
            igen_data.append(struct.pack("<Hh", 53, sample_id))

    def _build_presets_chunk(self):
        """
        Builds the preset-related chunks.
        """
        phdr_data, pbag_data, pmod_data, pgen_data = [], [], [], []
        inst_name_to_id = {inst["name"]: i for i, inst in enumerate(self.instruments)}

        for preset in self.presets:
            name = preset["name"][:20].ljust(20, "\x00").encode("ascii")
            bag_ndx = len(pbag_data)
            phdr_record = struct.pack(
                "<20sHHHIII",
                name,
                preset["preset_number"],
                preset["bank"],
                bag_ndx,
                preset.get("library", 0),
                preset.get("genre", 0),
                preset.get("morphology", 0)
            )
            phdr_data.append(phdr_record)

            # Process zones
            for zone in preset["zones"]:
                gen_ndx, mod_ndx = len(pgen_data), len(pmod_data)
                pbag_data.append(struct.pack("<HH", gen_ndx, mod_ndx))

                # Add modulators
                if "modulators" in zone:
                    for mod in zone["modulators"]:
                        pmod_data.append(struct.pack(
                            "<HHhHH",
                            mod["src_oper"],
                            mod["dest_oper"],
                            mod["amount"],
                            mod["amt_src_oper"],
                            mod["trans_oper"]
                        ))

                # Add generators
                self._add_preset_generators(pgen_data, zone["generators"], inst_name_to_id)

        # Terminators
        phdr_data.append(b"EOP".ljust(20, b"\x00") + struct.pack("<HHHIII", 0, 0, len(pbag_data), 0, 0, 0))
        pbag_data.append(struct.pack("<HH", len(pgen_data), len(pmod_data)))
        pmod_data.append(struct.pack("<HHhHH", 0, 0, 0, 0, 0))
        pgen_data.append(struct.pack("<Hh", 0, 0))

        return phdr_data, pbag_data, pmod_data, pgen_data

    def _add_preset_generators(self, pgen_data, generators, inst_name_to_id):
        """
        Adds preset generators.
        """
        # keyRange and velRange first
        if "keyRange" in generators:
            lo, hi = map(int, generators["keyRange"].split("-"))
            pgen_data.append(struct.pack("<Hh", 43, lo | (hi << 8)))
        if "velRange" in generators:
            lo, hi = map(int, generators["velRange"].split("-"))
            pgen_data.append(struct.pack("<Hh", 44, lo | (hi << 8)))

        # Other generators
        for gen_name, gen_value in generators.items():
            if gen_name in ["keyRange", "velRange", "instrument"]:
                continue
            gen_id = GENERATOR_IDS.get(gen_name)
            if gen_id:
                pgen_data.append(struct.pack("<Hh", gen_id, gen_value))

        # instrument last
        if "instrument" in generators:
            inst_ref = generators["instrument"]
            inst_id = inst_name_to_id.get(inst_ref, 0)
            if inst_id == 0 and inst_ref not in inst_name_to_id:
                print(f"Warning: Instrument \"{inst_ref}\" not found in mapping")
            pgen_data.append(struct.pack("<Hh", 41, inst_id))


def main():
    """
    Main function to run the SF2 compiler.
    """
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <input_directory> <output.sf2>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_sf2 = sys.argv[2]

    if not os.path.exists(input_dir):
        print(f"Error: Directory not found - {input_dir}")
        sys.exit(1)

    compiler = SF2Compiler(input_dir, output_sf2)
    compiler.compile()


if __name__ == "__main__":
    main()
