# Copyright (c) 2025 bakajikara
#
# This file is licensed under the MIT License (MIT).
# See the LICENSE file in the project root for the full license text.

"""
SoundFont Compiler - Rebuilds a SoundFont file from an expanded directory structure.

This tool generates a SoundFont file from the following structure:
- info.json: Metadata
- samples/: Audio files (FLAC, WAV, etc.)
- instruments/: Instrument definitions (JSON)
- presets/: Preset definitions (JSON)
"""

import os
import sys
import json
import struct
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod

try:
    import soundfile as sf
except ImportError:
    print("Error: soundfile library is required.")
    print("Install it with: pip install soundfile")
    sys.exit(1)

from .constants import GENERATOR_IDS, SF_SAMPLETYPE_VORBIS
from .riff import make_chunk, make_list_chunk, make_zstr

# ffmpeg is only required for SF3 compilation from non-OGG sources
# Import it conditionally when needed
_ffmpeg_available = None


def _check_ffmpeg():
    """
    Checks if ffmpeg-python is available.
    Returns True if available, False otherwise.
    """
    global _ffmpeg_available
    if _ffmpeg_available is None:
        try:
            import ffmpeg
            _ffmpeg_available = True
        except ImportError:
            _ffmpeg_available = False
    return _ffmpeg_available


class SoundFontCompiler(ABC):
    """
    Compiles a SoundFont file from an expanded directory structure.
    """

    def __new__(cls, input_dir, output_sf, quality=None):
        """
        Factory method to create a SoundFontCompiler instance.

        Args:
            input_dir: The input directory path.
            output_sf: The output SoundFont file path.
            quality: Ogg Vorbis quality (0.0-1.0, only for SF3). Default is 0.8.
        """
        # Determine subclass based on output file extension
        ext = Path(output_sf).suffix.lower()
        if ext == ".sf3":
            instance = super().__new__(_SF3Compiler)
        else:
            instance = super().__new__(_SF2Compiler)

        return instance

    def __init__(self, input_dir, output_sf, quality=None):
        """
        Initializes the SoundFont Compiler.

        Args:
            input_dir: The input directory path.
            output_sf: The output SoundFont file path.
            quality: Ogg Vorbis quality (0.0-1.0, only for SF3). Default is 0.8.
        """
        self.input_dir = Path(input_dir)
        self.output_sf = output_sf

        # Data storage
        self.bank_info = {}
        self.samples = []
        self.instruments = []
        self.presets = []

        # Parsed SoundFont version
        self.sf_major_version = 0
        self.sf_minor_version = 0

    @abstractmethod
    def _get_sample_padding_bytes(self):
        """
        Returns the padding bytes to add between samples for both smpl and sm24.

        Returns:
            A tuple of (smpl_padding_bytes, sm24_padding_bytes).
        """
        pass

    @abstractmethod
    def _update_sample_offset(self, current_offset, num_samples, data_length):
        """
        Updates the sample offset after writing a sample.

        Args:
            current_offset: Current offset in sample units (SF2) or bytes (SF3).
            num_samples: Number of samples written.
            data_length: Length of the data written in bytes.

        Returns:
            Updated offset.
        """
        pass

    @abstractmethod
    def _get_default_soundfont_version(self):
        """
        Returns the default SoundFont version string for this format.
        SF2 returns "2.04", SF3 returns "3.01".
        """
        pass

    @abstractmethod
    def _build_sample_header_record(self, sample, idx):
        """
        Builds a single sample header record.
        SF2 and SF3 have different offset and type handling.

        Args:
            sample: Sample dictionary with metadata and calculated positions.
            idx: Sample index.

        Returns:
            46-byte sample header record.
        """
        pass

    def compile(self):
        """
        Generates an SF2 file from the directory structure.
        """
        print(f"Compiling from: {self.input_dir}")

        # Load all parts
        self._load_bank_info()

        # Check for duplicates across all subdirectories before loading
        self._check_all_duplicates()

        self._load_samples()
        self._fix_stereo_links()  # Fix stereo sample links after loading
        self._load_instruments()
        self._load_presets()

        # Generate the SoundFont file
        print(f"Writing SoundFont file: {self.output_sf}")
        self._write_soundfont_file()

        print("Compilation complete!")

    def _fix_stereo_links(self):
        """
        Fixes stereo sample links after parallel loading.
        Stereo pairs need to reference each other by index.
        """
        # Group stereo samples by name
        stereo_groups = {}
        for idx, sample in enumerate(self.samples):
            if sample.get("_is_stereo"):
                key = sample["sample_name"]
                if key not in stereo_groups:
                    stereo_groups[key] = []
                stereo_groups[key].append(idx)

        # Set mutual links for each stereo pair
        for indices in stereo_groups.values():
            if len(indices) == 2:
                idx1, idx2 = indices[0], indices[1]
                sample1, sample2 = self.samples[idx1], self.samples[idx2]

                # Verify channel consistency
                if (
                    (sample1.get("_channel") == "left" and sample2.get("_channel") == "right") or
                    (sample1.get("_channel") == "right" and sample2.get("_channel") == "left")
                ):
                    # Set mutual links
                    sample1["sample_link"] = idx2
                    sample2["sample_link"] = idx1
                else:
                    print(f"  Warning: Channel mismatch in stereo pair for sample \"{sample1["sample_name"]}\" and \"{sample2["sample_name"]}\".")

            else:
                print(f"  Warning: Incomplete stereo pair for sample(s) \"{"\", \"".join(self.samples[idx]["sample_name"] for idx in indices)}\".")

    def _load_bank_info(self):
        """
        Loads info.json.
        """
        info_path = self.input_dir / "info.json"

        if not info_path.exists():
            raise FileNotFoundError(f"info.json not found in {self.input_dir}")

        with open(info_path, "r", encoding="utf-8") as f:
            self.bank_info = json.load(f)

        # Parse and store the version for later use
        # Use format-specific major version (2 for SF2, 3 for SF3)
        # and minor version from info.json
        default_version = self._get_default_soundfont_version()
        major, _ = map(int, default_version.split("."))
        version = self.bank_info.get("version", default_version)
        _, minor = map(int, version.split("."))
        self.sf_major_version = major
        self.sf_minor_version = minor

        print(f"  Loaded: info.json")

    def _check_all_duplicates(self):
        """
        Checks for duplicate filenames across all subdirectories (samples, instruments, presets).
        Reports all duplicates at once before proceeding with compilation.
        For samples, checks both JSON and audio files (.flac, .wav, .ogg, .oga, .mp3).
        """
        all_duplicates = []

        for subdir_name in ["samples", "instruments", "presets"]:
            subdir = self.input_dir / subdir_name
            if not subdir.exists():
                continue

            # For samples, check both JSON and audio files
            if subdir_name == "samples":
                file_patterns = ["*.json", "*.flac", "*.wav", "*.ogg", "*.oga", "*.mp3"]
            else:
                file_patterns = ["*.json"]

            seen_files = {}
            for pattern in file_patterns:
                files = sorted(subdir.rglob(pattern))
                for file_path in files:
                    if file_path.name in seen_files:
                        all_duplicates.append((
                            subdir_name,
                            file_path.name,
                            seen_files[file_path.name],
                            file_path
                        ))
                    else:
                        seen_files[file_path.name] = file_path

        # If duplicates were found, report all of them at once
        if all_duplicates:
            error_msg = "Duplicate filenames found in the input directory:\n\n"
            for subdir_name, filename, path1, path2 in all_duplicates:
                error_msg += f"Duplicate in {subdir_name}/: \"{filename}\"\n"
                error_msg += f"  1. {path1.relative_to(self.input_dir)}\n"
                error_msg += f"  2. {path2.relative_to(self.input_dir)}\n\n"
            error_msg += "Please reorganize your files to avoid duplicate names within the same subdirectory."
            raise ValueError(error_msg)

    def _load_samples(self):
        """
        Loads sample metadata from the samples directory.
        PCM data is read later for performance.
        """
        samples_dir = self.input_dir / "samples"

        if not samples_dir.exists():
            raise FileNotFoundError(f"samples directory not found in {self.input_dir}")

        # Load JSON files from samples directory and subdirectories
        json_files = sorted(samples_dir.rglob("*.json"))

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
        for ext in [".flac", ".wav", ".ogg", ".oga", "mp3"]:  # Supported audio formats; keep in sync with decompiler _cleanup_unwritten_files
            candidate = json_path.with_suffix(ext)
            if candidate.exists():
                audio_path = candidate
                break

        if audio_path is None:
            raise FileNotFoundError(f"Audio file not found for: {json_path}")

        sample_type = metadata.get("sample_type", "mono")

        if sample_type == "stereo":
            return self._create_combined_stereo_sample_entries(metadata, audio_path)
        else:
            return [self._create_single_channel_entry(metadata, audio_path)]

    def _create_combined_stereo_sample_entries(self, metadata, audio_path):
        """
        Creates stereo sample entries (left and right).
        Note: start/end will be calculated dynamically during compilation.
        """
        # Stereo sample: split into two samples
        start_loop = metadata.get("start_loop", 0)
        end_loop = metadata.get("end_loop", 0)
        original_key = metadata.get("original_key", 60)
        correction = metadata.get("correction", 0)

        # Left channel sample
        left_sample = {
            "sample_name": metadata["sample_name"],
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

    def _create_single_channel_entry(self, metadata, audio_path):
        """
        Creates a mono sample entry, or an entry for one channel of a split stereo pair.
        Note: start/end will be calculated dynamically during compilation.
        """
        sample_type = metadata.get("sample_type", "mono")

        entry = {
            "sample_name": metadata["sample_name"],
            "start_loop": metadata.get("start_loop", 0),
            "end_loop": metadata.get("end_loop", 0),
            "original_key": metadata.get("original_key", 60),
            "correction": metadata.get("correction", 0),
            "sample_link": 0,  # Will be set later by _fix_stereo_links
            "_audio_path": audio_path,
        }

        # Set sample_type and channel info
        if sample_type == "stereo_left":
            entry["sample_type"] = 4
            entry["_channel"] = "left"
            entry["_is_stereo"] = True
        elif sample_type == "stereo_right":
            entry["sample_type"] = 2
            entry["_channel"] = "right"
            entry["_is_stereo"] = True
        else:  # "mono"
            entry["sample_type"] = 1
            entry["_channel"] = None
            entry["_is_stereo"] = False

        return entry

    @abstractmethod
    def _read_audio_data(self, audio_path, channel=None):
        """
        Reads audio data from a file when needed.

        Args:
            audio_path: The path to the audio file.
            channel: "left", "right", or None (for mono).

        Returns:
            Tuple of (audio_data_bytes, sample_rate, num_samples)
        """
        pass

    @abstractmethod
    def _calculate_sample_positions(self, sample, current_offset, data_length):
        """
        Calculates absolute sample positions in the smpl chunk.

        Args:
            sample: The sample dictionary to update.
            current_offset: The current byte offset in the smpl chunk.
            data_length: The length of the audio data in bytes.
        """
        pass

    def _load_instruments(self):
        """
        Loads JSON files from the instruments directory.
        """
        instruments_dir = self.input_dir / "instruments"

        if not instruments_dir.exists():
            raise FileNotFoundError(f"instruments directory not found in {self.input_dir}")

        json_files = sorted(instruments_dir.rglob("*.json"))

        # Process in parallel with progress display, preserving order
        with ThreadPoolExecutor() as executor:
            # Submit all tasks and map futures to their index
            future_to_index = {
                executor.submit(self._load_json_file, json_path): idx
                for idx, json_path in enumerate(json_files)
            }

            # Collect results with their original indices
            results_with_index = []
            for future in as_completed(future_to_index):
                try:
                    data = future.result()
                    idx = future_to_index[future]
                    results_with_index.append((idx, data))
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

        json_files = sorted(presets_dir.rglob("*.json"))

        # Process in parallel with progress display, preserving order
        with ThreadPoolExecutor() as executor:
            # Submit all tasks and map futures to their index
            future_to_index = {
                executor.submit(self._load_json_file, json_path): idx
                for idx, json_path in enumerate(json_files)
            }

            # Collect results with their original indices
            results_with_index = []
            for future in as_completed(future_to_index):
                try:
                    data = future.result()
                    idx = future_to_index[future]
                    results_with_index.append((idx, data))
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

    def _write_soundfont_file(self):
        """
        Writes the SoundFont file.
        """
        with open(self.output_sf, "wb") as f:
            # RIFF header (size updated later)
            f.write(b"RIFF")
            riff_size_pos = f.tell()
            f.write(struct.pack("<I", 0))
            f.write(b"sfbk")

            # INFO-list
            info_chunk = self._build_info_chunk()
            print("  Writing INFO chunk...")
            f.write(info_chunk)

            # sdta-list
            sdta_chunk = self._build_sdta_chunk()
            print("  Writing sample data...")
            f.write(sdta_chunk)

            # pdta-list
            pdta_chunk = self._build_pdta_chunk()
            print("  Writing pdta chunk...")
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
        ifil_data = struct.pack("<HH", self.sf_major_version, self.sf_minor_version)
        data += make_chunk(b"ifil", ifil_data)

        # Sound engine (required, second)
        sound_engine = self.bank_info.get("sound_engine", "EMU8000")
        data += make_chunk(b"isng", make_zstr(sound_engine))

        # Bank name (required, third)
        bank_name = self.bank_info.get("bank_name", "Untitled")
        data += make_chunk(b"INAM", make_zstr(bank_name))

        # Output in SoundFont spec order (same as original files)
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

    def _build_sdta_chunk(self):
        """
        Builds the complete LIST sdta chunk from the raw sample data buffers.

        Returns:
            The complete LIST sdta chunk as a bytes object.
        """
        # Trigger parallel pre-processing of all audio samples
        audio_cache = self._preprocess_samples_parallel()

        # Build the raw sample data buffers (smpl and sm24)
        final_smpl_data_bytes, final_sm24_data_bytes = self._build_sample_data_buffers(audio_cache)

        sdta_content = b""

        # Create the smpl sub-chunk
        sdta_content += make_chunk(b"smpl", final_smpl_data_bytes)

        # Create the sm24 sub-chunk only if there is 24-bit data and version supports it
        if any(final_sm24_data_bytes):
            if self.sf_major_version == 2 and self.sf_minor_version >= 4:
                sdta_content += make_chunk(b"sm24", final_sm24_data_bytes)
            else:
                # This case occurs if there is 24-bit data but the target version is < 2.04
                print(f"  Warning: 24-bit sample data found, but target version is {self.sf_major_version}.{self.sf_minor_version:02d}. "
                      "The sm24 chunk will NOT be written, and samples will be truncated to 16-bit.")

        # Wrap everything in a LIST chunk
        return make_list_chunk(b"sdta", sdta_content)

    def _preprocess_samples_parallel(self):
        """
        Pre-processes all samples in parallel (encoding/reading audio data).
        Returns a list of (audio_data, sample_rate, num_samples) tuples.
        """
        total_samples = len(self.samples)
        print(f"  Processing {total_samples} samples...")

        audio_cache = [None] * total_samples

        with ThreadPoolExecutor() as executor:
            # Submit all tasks and map futures to their index
            future_to_index = {
                executor.submit(
                    self._read_audio_data,
                    sample["_audio_path"],
                    sample.get("_channel")
                ): idx
                for idx, sample in enumerate(self.samples)
            }

            # Collect results with their original indices
            for completed, future in enumerate(as_completed(future_to_index), 1):
                try:
                    result = future.result()
                    idx = future_to_index[future]
                    audio_cache[idx] = result

                    # Show progress inline
                    progress = (completed / total_samples) * 100
                    print(f"    Encoding: {completed}/{total_samples} ({progress:.1f}%)", end="\r")
                except Exception as e:
                    idx = future_to_index[future]
                    sample_name = self.samples[idx].get("sample_name", "unknown")
                    print(f"\n  ERROR processing sample {idx} ({sample_name}): {e}")
                    raise

        print()  # Newline after progress
        return audio_cache

    def _build_sample_data_buffers(self, audio_cache):
        """
        Iterates through cached samples to build the raw data buffers for smpl and sm24.
        This method also calculates and updates the absolute positions within self.samples.

        Args:
            audio_cache: A list of pre-processed audio data tuples.

        Returns:
            A tuple containing (final_smpl_data_bytes, final_sm24_data_bytes).
        """
        all_smpl_data_parts = []
        all_sm24_data_parts = []

        smpl_padding, sm24_padding = self._get_sample_padding_bytes()
        current_offset = 0

        print(f"  Building sample data buffers...")
        for idx, sample in enumerate(self.samples):
            smpl_data, sm24_data, sample_rate, num_samples = audio_cache[idx]

            sample["_sample_rate"] = sample_rate
            self._calculate_sample_positions(sample, current_offset, len(smpl_data))

            all_smpl_data_parts.append(smpl_data)
            all_smpl_data_parts.append(smpl_padding)

            if sm24_data is not None:
                all_sm24_data_parts.append(sm24_data)
                all_sm24_data_parts.append(sm24_padding)

            current_offset = self._update_sample_offset(current_offset, num_samples, len(smpl_data))

        return b"".join(all_smpl_data_parts), b"".join(all_sm24_data_parts)

    def _build_shdr_chunk(self):
        """
        Builds the shdr chunk.
        """
        shdr_data = []
        for idx, sample in enumerate(self.samples):
            # Use format-specific sample header building
            shdr_record = self._build_sample_header_record(sample, idx)
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


class _SF2Compiler(SoundFontCompiler):
    """
    Compiler for SF2 files (PCM audio).
    """

    # Padding between samples in sample units (minimum 46 samples or 92 bytes)
    SAMPLE_PADDING = 46

    def _get_sample_padding_bytes(self):
        """
        Returns PCM padding bytes for SF2 (92 bytes for smpl, 46 bytes for sm24).
        """
        smpl_padding = b"\x00" * self.SAMPLE_PADDING * 2  # 16-bit samples
        sm24_padding = b"\x00" * self.SAMPLE_PADDING      # 8-bit counterparts
        return smpl_padding, sm24_padding

    def _update_sample_offset(self, current_offset, num_samples, data_length):
        """
        Updates the sample offset in sample units for SF2.
        """
        return current_offset + num_samples + self.SAMPLE_PADDING

    def _get_default_soundfont_version(self):
        """
        Returns the default SoundFont version for SF2 format.
        """
        return "2.04"

    def _build_sample_header_record(self, sample, idx):
        """
        Builds a sample header record for SF2 format.

        SF2 uses absolute positions from the start of the smpl chunk.
        Loop positions are also absolute offsets.
        Sample type is standard (mono=1, right=2, left=4).
        """
        # Dynamically generate sample name (add left/right suffixes)
        base_name = sample["sample_name"]
        if sample.get("_is_stereo"):
            channel = sample.get("_channel")
            name = f"{base_name}_L"[:20] if channel == "left" else f"{base_name}_R"[:20]
        else:
            name = base_name[:20]

        name_bytes = name.ljust(20, "\x00").encode("ascii")

        # Use absolute positions calculated earlier
        start = sample.get("_absolute_start", 0)
        end = sample.get("_absolute_end", 0)
        start_loop = sample.get("_absolute_start_loop", 0)
        end_loop = sample.get("_absolute_end_loop", 0)
        sample_rate = sample.get("_sample_rate", 44100)

        # SF2: standard sample type (no compression flag)
        sample_type = sample["sample_type"]

        return struct.pack(
            "<20sIIIIIBbHH",
            name_bytes,
            start,
            end,
            start_loop,
            end_loop,
            sample_rate,
            sample["original_key"],
            sample["correction"],
            sample["sample_link"],
            sample_type
        )

    def _read_audio_data(self, audio_path, channel=None):
        """
        Reads PCM data from an audio file (FLAC, WAV, etc.) when needed.

        Args:
            audio_path: The path to the audio file.
            channel: "left", "right", or None (for mono).

        Returns:
            Tuple of (pcm_data_bytes, sample_rate, num_samples)
        """
        try:
            # Read as int32 with soundfile
            data, samplerate = sf.read(audio_path, dtype="int32")

            # Separate channels
            if channel == "left":
                if len(data.shape) == 2:
                    data = data[:, 0]
            elif channel == "right":
                if len(data.shape) == 2:
                    data = data[:, 1]

            # Convert to 24-bit PCM
            data >>= 8

            # MSB 16 bits for the "smpl" chunk (as signed 16-bit)
            smpl_data = (data >> 8).astype("int16").tobytes()
            # LSB 8 bits for the "sm24" chunk (as unsigned 8-bit)
            sm24_data = (data & 0xFF).astype("uint8").tobytes()

            num_samples = len(smpl_data) // 2

            return smpl_data, sm24_data, samplerate, num_samples
        except Exception as e:
            raise ValueError(f"Failed to read audio file {audio_path}: {e}")

    def _calculate_sample_positions(self, sample, current_offset, data_length):
        """
        Calculates absolute sample positions for SF2.

        SF2 uses sample units (not bytes) for all positions.
        Loop positions are absolute offsets from the start of smpl chunk.

        Args:
            sample: The sample dictionary to update.
            current_offset: The current offset in sample units.
            data_length: The length of the audio data in bytes.
        """
        # Calculate sample length from data_length (16-bit samples = 2 bytes per sample)
        num_samples = data_length // 2

        # Use absolute positions calculated from current offset
        sample["_absolute_start"] = current_offset
        sample["_absolute_end"] = current_offset + num_samples
        sample["_absolute_start_loop"] = current_offset + sample["start_loop"]
        sample["_absolute_end_loop"] = current_offset + sample["end_loop"]


class _SF3Compiler(SoundFontCompiler):
    """
    Compiler for SF3 files (Ogg Vorbis audio).
    """

    def __init__(self, input_dir, output_sf, quality=None):
        """
        Initializes the SF3 Compiler.

        Args:
            input_dir: The input directory path.
            output_sf: The output SoundFont file path.
            quality: Ogg Vorbis quality (0.0-1.0). Default is 0.8.
        """
        super().__init__(input_dir, output_sf, quality)
        # Ogg Vorbis quality setting (0.0 to 1.0)
        if quality is not None:
            if not 0.0 <= quality <= 1.0:
                raise ValueError(f"Quality must be between 0.0 and 1.0, got {quality}")
            self.ogg_quality = quality
        else:
            self.ogg_quality = 0.8

    def _get_sample_padding_bytes(self):
        """
        Returns empty padding bytes for SF3, as no padding is required.
        """
        return b"", b""

    def _update_sample_offset(self, current_offset, num_samples, data_length):
        """
        Updates the sample offset in bytes for SF3.
        For Ogg Vorbis, offsets are in bytes, not sample units.
        """
        return current_offset + data_length

    def _get_default_soundfont_version(self):
        """
        Returns the default SoundFont version for SF3 format.
        """
        return "3.01"

    def _build_sample_header_record(self, sample, idx):
        """
        Builds a sample header record for SF3 format.

        SF3 differences:
        - Start/end positions are absolute BYTE offsets (not sample units)
        - Loop positions are RELATIVE to the sample start (not absolute)
        - Sample type has 0x0010 flag added to indicate Ogg Vorbis compression
        """
        # Dynamically generate sample name (add left/right suffixes)
        base_name = sample["sample_name"]
        if sample.get("_is_stereo"):
            channel = sample.get("_channel")
            name = f"{base_name}_L"[:20] if channel == "left" else f"{base_name}_R"[:20]
        else:
            name = base_name[:20]

        name_bytes = name.ljust(20, "\x00").encode("ascii")

        # SF3: Start/end are absolute byte offsets in smpl chunk
        start = sample.get("_absolute_start", 0)
        end = sample.get("_absolute_end", 0)

        # SF3: Loop positions are RELATIVE to sample start
        # Calculate relative offsets from the original metadata
        start_loop = sample.get("start_loop", 0)
        end_loop = sample.get("end_loop", 0)

        sample_rate = sample.get("_sample_rate", 44100)

        # SF3: Add Vorbis compression flag to sample type
        base_sample_type = sample["sample_type"]
        sample_type = base_sample_type | SF_SAMPLETYPE_VORBIS

        return struct.pack(
            "<20sIIIIIBbHH",
            name_bytes,
            start,
            end,
            start_loop,
            end_loop,
            sample_rate,
            sample["original_key"],
            sample["correction"],
            sample["sample_link"],
            sample_type
        )

    def _read_audio_data(self, audio_path, channel=None):
        """
        Reads Ogg Vorbis data from an audio file when needed.

        Args:
            audio_path: The path to the audio file.
            channel: "left", "right", or None (for mono).

        Returns:
            Tuple of (ogg_data_bytes, sample_rate, num_samples)
        """
        try:
            # For SF3, we expect .ogg files
            if audio_path.suffix.lower() == ".ogg":
                # Read raw Ogg Vorbis data directly
                with open(audio_path, "rb") as f:
                    ogg_data = f.read()

                # Get metadata using soundfile
                info = sf.info(audio_path)
                sample_rate = info.samplerate
                num_samples = info.frames

                return ogg_data, None, sample_rate, num_samples
            else:
                # If not .ogg, read as PCM and encode to Ogg Vorbis
                ogg_data, samplerate, num_samples = self._encode_to_ogg_vorbis(audio_path, channel)
                return ogg_data, None, samplerate, num_samples
        except Exception as e:
            raise ValueError(f"Failed to read audio file {audio_path}: {e}")

    def _calculate_sample_positions(self, sample, current_offset, data_length):
        """
        Calculates absolute sample positions for SF3.

        SF3 uses byte offsets for start/end positions.
        Loop positions are relative to the sample start (already stored in metadata).

        Args:
            sample: The sample dictionary to update.
            current_offset: The current offset in bytes.
            data_length: The length of the audio data in bytes.
        """
        # SF3: start and end are absolute byte offsets in smpl chunk
        sample["_absolute_start"] = current_offset
        sample["_absolute_end"] = current_offset + data_length

        # Loop positions remain as-is (relative to sample start)
        # They are already stored in the metadata and will be used directly

    def _encode_to_ogg_vorbis(self, audio_path, channel=None):
        """
        Encodes an audio file to Ogg Vorbis format using ffmpeg.

        Args:
            audio_path: The path to the audio file.
            channel: "left", "right", or None (for mono).

        Returns:
            Tuple of (ogg_data_bytes, sample_rate, num_samples)
        """
        # Check if ffmpeg is available
        if not _check_ffmpeg():
            raise ImportError(
                "ffmpeg-python library is required for encoding non-OGG files to Ogg Vorbis.\n"
                "Install it with: pip install ffmpeg-python\n"
                "FFmpeg must also be installed on your system."
            )

        import ffmpeg

        # Read audio data to get metadata and separate channels if needed
        data, samplerate = sf.read(audio_path, dtype="float64")

        # Separate channels if needed
        if channel == "left":
            if len(data.shape) == 2:
                data = data[:, 0]
        elif channel == "right":
            if len(data.shape) == 2:
                data = data[:, 1]

        num_samples = len(data)

        # Convert float64 numpy array to bytes (no conversion needed)
        audio_bytes = data.tobytes()

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_output:
            tmp_output_path = tmp_output.name

        try:
            # Convert to Ogg Vorbis using ffmpeg with pipe input
            # Quality setting: 0-10 scale, convert from 0.0-1.0 to 0-10
            quality = int(self.ogg_quality * 10)

            process = (
                ffmpeg
                .input("pipe:", format="f64le", acodec="pcm_f64le", ar=str(samplerate), ac=1)
                .output(tmp_output_path, acodec="libvorbis", **{"q:a": quality, "threads": 1})
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True)
            )

            # Write float32 audio data to ffmpeg stdin and get output
            stdout, stderr = process.communicate(input=audio_bytes)

            if process.returncode != 0:
                stderr_text = stderr.decode() if stderr else "Unknown error"
                raise ValueError(f"ffmpeg failed with return code {process.returncode}: {stderr_text}")

            # Read back the encoded Ogg Vorbis data
            with open(tmp_output_path, "rb") as f:
                ogg_data = f.read()

            return ogg_data, samplerate, num_samples
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to encode audio with ffmpeg: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_output_path):
                os.unlink(tmp_output_path)


def main():
    """
    Main function to run the SoundFont compiler.
    """
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <input_directory> <output_file>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_sf = sys.argv[2]

    if not os.path.exists(input_dir):
        print(f"Error: Directory not found - {input_dir}")
        sys.exit(1)

    compiler = SoundFontCompiler(input_dir, output_sf)
    compiler.compile()


if __name__ == "__main__":
    main()
