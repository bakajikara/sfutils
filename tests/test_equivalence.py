#!/usr/bin/env python3
"""
SoundFont Equivalence Test - verify two SF2/SF3 files are practically equivalent

This test ignores:
- Order of samples, instruments, presets
- Minor differences in internal sample/instrument names
- Differences in IDs/offsets due to ordering

This test detects:
- Differences in outward-facing (user-visible) preset names, etc.
- Inconsistencies in references (checks correctness even if order differs)
- Differences in audio data (PCM for SF2, Ogg Vorbis for SF3)
- Differences in generator/modulator parameter values
"""

import sys
import struct
import hashlib
from pathlib import Path
from collections import defaultdict


class SoundFontEquivalenceChecker:
    """Check practical equivalence of two SoundFont files (SF2 or SF3)"""

    def __init__(self, file1_path, file2_path):
        self.file1_path = Path(file1_path)
        self.file2_path = Path(file2_path)
        self.errors = []
        self.warnings = []

        self.sample_mapping = {}
        self.samples1_hashes = {}
        self.samples2_hashes = {}

    def check(self):
        """Run the equivalence check"""
        print("=" * 80)
        print("SoundFont Equivalence Test")
        print("=" * 80)
        print(f"File 1: {self.file1_path}")
        print(f"File 2: {self.file2_path}")
        print()

        # Parse both files
        print("Parsing files...")
        data1 = self._parse_sf(self.file1_path)
        data2 = self._parse_sf(self.file2_path)

        if data1 is None or data2 is None:
            print("❌ Failed to parse one or both files")
            return False

        print(f"  File 1: Version {data1["info"].get("version", "N/A")}, {len(data1["samples"])} samples, {len(data1["instruments"])} instruments, {len(data1["presets"])} presets")
        print(f"  File 2: Version {data2["info"].get("version", "N/A")}, {len(data2["samples"])} samples, {len(data2["instruments"])} instruments, {len(data2["presets"])} presets")
        print()

        # Determine file type from version string
        is_sf3 = data1["info"].get("version", "2.").startswith("3")

        # Check each element
        print("Checking equivalence...")
        print()

        self._check_info(data1["info"], data2["info"])
        self._check_samples(data1["samples"], data2["samples"], data1["smpl_data"], data2["smpl_data"], data1["sm24_data"], data2["sm24_data"], is_sf3)
        self._check_instruments(data1["instruments"], data2["instruments"], data1["samples"], data2["samples"])
        self._check_presets(data1["presets"], data2["presets"], data1["instruments"], data2["instruments"])

        # Display results
        print()
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)

        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
            print()
            print("Files are NOT equivalent")
            return False
        else:
            print("\n✅ Files are equivalent!")
            print("   (no practical differences detected)")
            return True

    def _parse_sf(self, filepath):
        """Parse an SF2/SF3 file"""
        try:
            # The parsing logic is largely the same for the structures we care about.
            # The difference is in how we interpret the data, which is handled later.
            with open(filepath, "rb") as f:
                # RIFF header
                riff_id = f.read(4)
                if riff_id != b"RIFF":
                    self.errors.append(f"{filepath}: Not a RIFF file")
                    return None

                file_size = struct.unpack("<I", f.read(4))[0]
                form_type = f.read(4)
                if form_type != b"sfbk":
                    self.errors.append(f"{filepath}: Not a SoundFont file")
                    return None

                # Data structure
                data = {
                    "info": {},
                    "smpl_data": b"",
                    "sm24_data": b"",
                    "samples": [],
                    "instruments": [],
                    "presets": [],
                    "pdta_raw": {}
                }

                # Read main LIST chunks
                file_end = f.tell() + file_size - 4
                while f.tell() < file_end:
                    chunk_id = f.read(4)
                    if len(chunk_id) < 4:
                        break
                    chunk_size = struct.unpack("<I", f.read(4))[0]

                    if chunk_id == b"LIST":
                        list_type = f.read(4)
                        list_end = f.tell() + chunk_size - 4
                        if list_type == b"INFO":
                            data["info"] = self._parse_info_chunk(f, list_end)
                        elif list_type == b"sdta":
                            data["smpl_data"], data["sm24_data"] = self._parse_sdta_chunk(f, list_end)
                        elif list_type == b"pdta":
                            data["pdta_raw"] = self._parse_pdta_chunk(f, list_end)
                        f.seek(list_end)
                    else:
                        f.seek(chunk_size, 1)

                    if chunk_size % 2:
                        f.seek(1, 1)

                # Structure pdta data
                self._structure_pdta(data)
                return data

        except Exception as e:
            self.errors.append(f"Failed to parse {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _parse_info_chunk(self, f, chunk_end):
        """Parse the INFO chunk"""
        info = {}
        while f.tell() < chunk_end:
            sub_id = f.read(4)
            if len(sub_id) < 4:
                break
            sub_size = struct.unpack("<I", f.read(4))[0]
            data = f.read(sub_size)
            if sub_size % 2:
                f.read(1)

            sub_id_str = sub_id.decode("ascii", errors="ignore")
            if sub_id == b"ifil":
                major, minor = struct.unpack("<HH", data)
                info["version"] = f"{major}.{minor:02d}"
            elif sub_id == b"INAM":
                info["bank_name"] = data.decode("ascii", errors="ignore").rstrip("\x00")
            elif sub_id == b"isng":
                info["sound_engine"] = data.decode("ascii", errors="ignore").rstrip("\x00")
            elif sub_id == b"ICOP":
                info["copyright"] = data.decode("ascii", errors="ignore").rstrip("\x00")

        return info

    def _parse_sdta_chunk(self, f, chunk_end):
        """Parse the sdta chunk (sample data)"""
        smpl_data = b""
        sm24_data = b""
        while f.tell() < chunk_end:
            sub_id = f.read(4)
            if len(sub_id) < 4:
                break
            sub_size = struct.unpack("<I", f.read(4))[0]

            if sub_id == b"smpl":
                smpl_data = f.read(sub_size)
            elif sub_id == b"sm24":
                sm24_data = f.read(sub_size)
            else:
                f.read(sub_size)

            if sub_size % 2:
                f.read(1)

        return smpl_data, sm24_data

    def _parse_pdta_chunk(self, f, chunk_end):
        """Parse the pdta chunk"""
        pdta = {}
        while f.tell() < chunk_end:
            sub_id = f.read(4)
            if len(sub_id) < 4:
                break
            sub_size = struct.unpack("<I", f.read(4))[0]
            data = f.read(sub_size)
            if sub_size % 2:
                f.read(1)

            sub_id_str = sub_id.decode("ascii", errors="ignore")
            pdta[sub_id_str] = data

        return pdta

    def _structure_pdta(self, data):
        """Structure raw pdta data"""
        pdta = data["pdta_raw"]

        # Sample headers
        shdr_data = pdta.get("shdr", b"")
        for i in range(0, len(shdr_data), 46):
            if i + 46 > len(shdr_data):
                break
            chunk = shdr_data[i:i + 46]
            name = chunk[0:20].decode("ascii", errors="ignore").rstrip("\x00")
            if name == "EOS" or not name:
                break

            values = struct.unpack("<IIIIIBbHH", chunk[20:46])
            data["samples"].append({
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

        # Instrument headers and bags
        inst_data = pdta.get("inst", b"")
        ibag_data = pdta.get("ibag", b"")
        igen_data = pdta.get("igen", b"")
        imod_data = pdta.get("imod", b"")

        inst_headers = []
        for i in range(0, len(inst_data), 22):
            if i + 22 > len(inst_data):
                break
            name = inst_data[i:i + 20].decode("ascii", errors="ignore").rstrip("\x00")
            if name == "EOI" or not name:
                break
            bag_ndx = struct.unpack("<H", inst_data[i + 20:i + 22])[0]
            inst_headers.append({"name": name, "bag_ndx": bag_ndx})

        ibags = []
        for i in range(0, len(ibag_data), 4):
            if i + 4 > len(ibag_data):
                break
            gen_ndx, mod_ndx = struct.unpack("<HH", ibag_data[i:i + 4])
            ibags.append({"gen_ndx": gen_ndx, "mod_ndx": mod_ndx})

        igens = []
        for i in range(0, len(igen_data), 4):
            if i + 4 > len(igen_data):
                break
            oper, amount = struct.unpack("<Hh", igen_data[i:i + 4])
            igens.append({"oper": oper, "amount": amount})

        imods = []
        for i in range(0, len(imod_data), 10):
            if i + 10 > len(imod_data):
                break
            values = struct.unpack("<HHhHH", imod_data[i:i + 10])
            imods.append({
                "src_oper": values[0],
                "dest_oper": values[1],
                "amount": values[2],
                "amt_src_oper": values[3],
                "trans_oper": values[4]
            })

        # Structure instruments
        for idx, header in enumerate(inst_headers):
            bag_start = header["bag_ndx"]
            bag_end = inst_headers[idx + 1]["bag_ndx"] if idx + 1 < len(inst_headers) else len(ibags) - 1

            zones = []
            for bag_idx in range(bag_start, bag_end):
                if bag_idx >= len(ibags):
                    break
                bag = ibags[bag_idx]

                gen_start = bag["gen_ndx"]
                gen_end = ibags[bag_idx + 1]["gen_ndx"] if bag_idx + 1 < len(ibags) else len(igens)

                mod_start = bag["mod_ndx"]
                mod_end = ibags[bag_idx + 1]["mod_ndx"] if bag_idx + 1 < len(ibags) else len(imods)

                zone_gens = igens[gen_start:gen_end] if gen_start < len(igens) else []
                zone_mods = imods[mod_start:mod_end] if mod_start < len(imods) else []

                zones.append({
                    "generators": zone_gens,
                    "modulators": zone_mods
                })

            data["instruments"].append({
                "name": header["name"],
                "zones": zones
            })

        # Preset headers and bags
        phdr_data = pdta.get("phdr", b"")
        pbag_data = pdta.get("pbag", b"")
        pgen_data = pdta.get("pgen", b"")
        pmod_data = pdta.get("pmod", b"")

        preset_headers = []
        for i in range(0, len(phdr_data), 38):
            if i + 38 > len(phdr_data):
                break
            chunk = phdr_data[i:i + 38]
            name = chunk[0:20].decode("ascii", errors="ignore").rstrip("\x00")
            if name == "EOP" or not name:
                break
            preset, bank, bag_ndx = struct.unpack("<HHH", chunk[20:26])
            library, genre, morphology = struct.unpack("<III", chunk[26:38])
            preset_headers.append({
                "name": name,
                "preset": preset,
                "bank": bank,
                "bag_ndx": bag_ndx,
                "library": library,
                "genre": genre,
                "morphology": morphology
            })

        pbags = []
        for i in range(0, len(pbag_data), 4):
            if i + 4 > len(pbag_data):
                break
            gen_ndx, mod_ndx = struct.unpack("<HH", pbag_data[i:i + 4])
            pbags.append({"gen_ndx": gen_ndx, "mod_ndx": mod_ndx})

        pgens = []
        for i in range(0, len(pgen_data), 4):
            if i + 4 > len(pgen_data):
                break
            oper, amount = struct.unpack("<Hh", pgen_data[i:i + 4])
            pgens.append({"oper": oper, "amount": amount})

        pmods = []
        for i in range(0, len(pmod_data), 10):
            if i + 10 > len(pmod_data):
                break
            values = struct.unpack("<HHhHH", pmod_data[i:i + 10])
            pmods.append({
                "src_oper": values[0],
                "dest_oper": values[1],
                "amount": values[2],
                "amt_src_oper": values[3],
                "trans_oper": values[4]
            })

        # Structure presets
        for idx, header in enumerate(preset_headers):
            bag_start = header["bag_ndx"]
            bag_end = preset_headers[idx + 1]["bag_ndx"] if idx + 1 < len(preset_headers) else len(pbags) - 1

            zones = []
            for bag_idx in range(bag_start, bag_end):
                if bag_idx >= len(pbags):
                    break
                bag = pbags[bag_idx]

                gen_start = bag["gen_ndx"]
                gen_end = pbags[bag_idx + 1]["gen_ndx"] if bag_idx + 1 < len(pbags) else len(pgens)

                mod_start = bag["mod_ndx"]
                mod_end = pbags[bag_idx + 1]["mod_ndx"] if bag_idx + 1 < len(pbags) else len(pmods)

                zone_gens = pgens[gen_start:gen_end] if gen_start < len(pgens) else []
                zone_mods = pmods[mod_start:mod_end] if mod_start < len(pmods) else []

                zones.append({
                    "generators": zone_gens,
                    "modulators": zone_mods
                })

            data["presets"].append({
                "name": header["name"],
                "preset": header["preset"],
                "bank": header["bank"],
                "library": header["library"],
                "genre": header["genre"],
                "morphology": header["morphology"],
                "zones": zones
            })

    def _check_info(self, info1, info2):
        """Compare INFO chunk (only outward-facing information)"""
        print("Checking INFO (metadata)...")

        # External information (user-visible)
        external_keys = ["bank_name", "version"]

        for key in external_keys:
            val1 = info1.get(key, "")
            val2 = info2.get(key, "")
            if val1 != val2:
                self.errors.append(f"INFO.{key}: \"{val1}\" != \"{val2}\"")

        # Other info only as warnings
        internal_keys = ["sound_engine", "copyright"]
        for key in internal_keys:
            val1 = info1.get(key, "")
            val2 = info2.get(key, "")
            if val1 != val2:
                self.warnings.append(f"INFO.{key}: \"{val1}\" != \"{val2}\" (internal info differs)")

        print("  ✓ INFO check complete")

    def _check_samples(self, samples1, samples2, smpl_data1, smpl_data2, sm24_data1, sm24_data2, is_sf3):
        """
        Check sample equivalence by first matching audio data, then comparing metadata field-by-field.
        This provides robust matching while also giving detailed error reports on mismatches.
        """
        print("Checking samples...")

        if len(samples1) != len(samples2):
            self.errors.append(f"Sample count mismatch: {len(samples1)} != {len(samples2)}")
            return

        def get_audio_data_and_hash(sample, smpl_data, sm24_data, is_sf3_mode):
            """Returns the raw audio data and its MD5 hash."""
            if is_sf3_mode:
                audio_data = smpl_data[sample["start"]:sample["end"]]
            else:  # SF2
                audio_data_16 = smpl_data[sample["start"] * 2:sample["end"] * 2]
                audio_data_24 = sm24_data[sample["start"]:sample["end"]] if sm24_data else b""
                audio_data = audio_data_16 + audio_data_24
            return audio_data, hashlib.md5(audio_data).hexdigest()

        # Step 1: Index all samples in file1 by their audio hash.
        # The value is a list to handle multiple samples with identical audio data.
        samples1_by_audio = defaultdict(list)
        for idx, s1 in enumerate(samples1):
            _, audio_hash = get_audio_data_and_hash(s1, smpl_data1, sm24_data1, is_sf3)
            samples1_by_audio[audio_hash].append((idx, s1))

        # Also create simple hash lookups for the instrument checker later
        self.samples1_hashes = {idx: get_audio_data_and_hash(s, smpl_data1, sm24_data1, is_sf3)[1] for idx, s in enumerate(samples1)}
        self.samples2_hashes = {idx: get_audio_data_and_hash(s, smpl_data2, sm24_data2, is_sf3)[1] for idx, s in enumerate(samples2)}

        # Step 2: Iterate through file2 samples and find a perfect match in file1.
        matched_count = 0
        for idx2, s2 in enumerate(samples2):
            _, audio_hash2 = get_audio_data_and_hash(s2, smpl_data2, sm24_data2, is_sf3)

            found_match = False
            if audio_hash2 in samples1_by_audio:
                # We have candidate samples with identical audio. Now check metadata.
                candidate_list = samples1_by_audio[audio_hash2]

                for i, (idx1, s1_candidate) in enumerate(candidate_list):
                    if self._are_sample_metadata_equal(s1_candidate, s2, is_sf3):
                        # Perfect match found!
                        self.sample_mapping[idx2] = idx1
                        # Remove the matched candidate so it can't be matched again
                        candidate_list.pop(i)
                        found_match = True
                        matched_count += 1
                        break  # Move to the next sample in file2

                if not found_match:
                    # Audio matched, but no candidate had matching metadata. Report the diff.
                    first_candidate_idx, first_candidate_s1 = samples1_by_audio[audio_hash2][0]
                    self.errors.append(f"Sample #{idx2} has matching audio with #{first_candidate_idx}, but metadata differs:")
                    self._report_metadata_diff(first_candidate_s1, s2, is_sf3)

            else:
                self.errors.append(f"Sample #{idx2} in file2 has audio data not found in file1.")

        # Step 3: Check if any samples from file1 were left unmatched.
        unmatched_count = sum(len(candidates) for candidates in samples1_by_audio.values())
        if unmatched_count > 0:
            self.errors.append(f"{unmatched_count} samples from file1 were not found in file2.")

        if not self.errors:
            print(f"  ✓ All {len(samples1)} samples matched successfully.")
            # Step 4: With a reliable mapping, verify the stereo links.
            self._verify_sample_links(samples1, samples2, self.sample_mapping)

    def _get_relative_loops(self, sample, is_sf3):
        """Returns a tuple of (start_loop_relative, end_loop_relative)."""
        if is_sf3:
            return sample["start_loop"], sample["end_loop"]
        else:  # SF2
            return sample["start_loop"] - sample["start"], sample["end_loop"] - sample["start"]

    def _are_sample_metadata_equal(self, s1, s2, is_sf3):
        """Returns True if all critical metadata fields are equal."""
        loops1 = self._get_relative_loops(s1, is_sf3)
        loops2 = self._get_relative_loops(s2, is_sf3)

        return (s1["sample_rate"] == s2["sample_rate"] and
                s1["original_key"] == s2["original_key"] and
                s1["correction"] == s2["correction"] and
                s1["sample_type"] == s2["sample_type"] and
                loops1 == loops2)

    def _report_metadata_diff(self, s1, s2, is_sf3):
        """Adds detailed error messages for each differing metadata field."""
        if s1["sample_rate"] != s2["sample_rate"]:
            self.errors.append(f"    - sample_rate: {s1["sample_rate"]} (file1) vs {s2["sample_rate"]} (file2)")
        if s1["original_key"] != s2["original_key"]:
            self.errors.append(f"    - original_key: {s1["original_key"]} (file1) vs {s2["original_key"]} (file2)")
        if s1["correction"] != s2["correction"]:
            self.errors.append(f"    - correction: {s1["correction"]} (file1) vs {s2["correction"]} (file2)")
        if s1["sample_type"] != s2["sample_type"]:
            self.errors.append(f"    - sample_type: {s1["sample_type"]} (file1) vs {s2["sample_type"]} (file2)")

        loops1 = self._get_relative_loops(s1, is_sf3)
        loops2 = self._get_relative_loops(s2, is_sf3)
        if loops1 != loops2:
            self.errors.append(f"    - relative_loops: {loops1} (file1) vs {loops2} (file2)")

    def _verify_sample_links(self, samples1, samples2, sample_mapping):
        """
        Verify that the linked partners of matched samples also match.
        This explicitly checks the integrity of stereo pairs.
        """
        print("  Verifying sample links for stereo pairs...")
        link_errors = 0

        # sample_mapping is {idx2: idx1}
        for idx2, idx1 in sample_mapping.items():
            sample1 = samples1[idx1]
            sample2 = samples2[idx2]

            # Check if this sample is part of a stereo pair
            is_stereo1 = sample1["sample_type"] & 6 and sample1["sample_link"] < len(samples1)
            is_stereo2 = sample2["sample_type"] & 6 and sample2["sample_link"] < len(samples2)

            # If both are stereo, check if their partners match
            if is_stereo1 and is_stereo2:
                linked_idx1 = sample1["sample_link"]
                linked_idx2 = sample2["sample_link"]

                # The linked partner of sample2 (linked_idx2) should map to
                # the linked partner of sample1 (linked_idx1) in our mapping.
                if sample_mapping.get(linked_idx2) != linked_idx1:
                    self.errors.append(f"Sample link mismatch for pair #{idx1} vs #{idx2}. "
                                       f"Partner {linked_idx1} does not correspond to {linked_idx2}.")
                    link_errors += 1

        if link_errors == 0:
            print("    ✓ Sample links are consistent.")

    def _check_instruments(self, instruments1, instruments2, samples1, samples2):
        """Compare instruments (ignore order)"""
        print("Checking instruments...")

        if len(instruments1) != len(instruments2):
            self.errors.append(f"Instrument count mismatch: {len(instruments1)} != {len(instruments2)}")

        # Map instruments by content
        def inst_signature(inst, sample_hashes):
            """Generate instrument signature (excluding name)"""
            sig = []
            for zone in inst["zones"]:
                zone_gens = []
                zone_mods = []

                for gen in zone["generators"]:
                    # For sampleID (oper=53), use the sample's actual PCM hash
                    if gen["oper"] == 53:  # sampleID
                        sample_idx = gen["amount"]
                        if sample_idx in sample_hashes:
                            # Use the sample hash value to uniquely identify
                            zone_gens.append((53, sample_hashes[sample_idx]))
                        else:
                            zone_gens.append((53, f"invalid_{sample_idx}"))
                    else:
                        zone_gens.append((gen["oper"], gen["amount"]))

                for mod in zone["modulators"]:
                    zone_mods.append((mod["src_oper"], mod["dest_oper"], mod["amount"], mod["amt_src_oper"], mod["trans_oper"]))

                # Sort generators and modulators separately, then combine
                sig.append((tuple(sorted(zone_gens)), tuple(sorted(zone_mods))))

            return tuple(sorted(sig))

        # Index instruments of file1
        inst1_by_sig = {}
        for idx, inst in enumerate(instruments1):
            sig = inst_signature(inst, self.samples1_hashes)
            if sig not in inst1_by_sig:
                inst1_by_sig[sig] = []
            inst1_by_sig[sig].append((idx, inst))

        # Match each instrument in file2
        matched = set()
        inst_mapping = {}  # file2_idx -> file1_idx

        for idx2, inst2 in enumerate(instruments2):
            sig2 = inst_signature(inst2, self.samples2_hashes)
            if sig2 in inst1_by_sig and inst1_by_sig[sig2]:
                idx1, inst1 = inst1_by_sig[sig2].pop(0)
                matched.add(idx1)
                inst_mapping[idx2] = idx1

                # Name differences are only warnings (internal names)
                if inst1["name"] != inst2["name"]:
                    self.warnings.append(f"Instrument name differs: \"{inst1["name"]}\" vs \"{inst2["name"]}\" (internal names)")

            else:
                self.errors.append(f"Instrument #{idx2} \"{inst2["name"]}\" in file2 not found in file1")

        # Check for unmatched instruments
        for sig, remaining in inst1_by_sig.items():
            for idx1, inst1 in remaining:
                if idx1 not in matched:
                    self.errors.append(f"Instrument #{idx1} \"{inst1["name"]}\" in file1 not found in file2")

        print(f"  ✓ {len(matched)}/{len(instruments1)} instruments matched")

        # Save instrument mapping
        self.inst_mapping = inst_mapping

    def _check_presets(self, presets1, presets2, instruments1, instruments2):
        """Compare presets (ignore order)"""
        print("Checking presets...")

        if len(presets1) != len(presets2):
            self.errors.append(f"Preset count mismatch: {len(presets1)} != {len(presets2)}")

        # Group presets by bank/preset numbers
        def group_by_bank_preset(presets):
            """Group by bank/preset numbers"""
            grouped = defaultdict(list)
            for preset in presets:
                key = (preset["bank"], preset["preset"])
                grouped[key].append(preset)
            return grouped

        presets1_grouped = group_by_bank_preset(presets1)
        presets2_grouped = group_by_bank_preset(presets2)

        # Verify bank/preset number sets match
        keys1 = set(presets1_grouped.keys())
        keys2 = set(presets2_grouped.keys())

        if keys1 != keys2:
            missing_in_2 = keys1 - keys2
            missing_in_1 = keys2 - keys1
            if missing_in_2:
                self.errors.append(f"Preset bank/preset numbers in file1 but not in file2: {missing_in_2}")
            if missing_in_1:
                self.errors.append(f"Preset bank/preset numbers in file2 but not in file1: {missing_in_1}")

        # Compare presets within each bank/preset number
        matched_count = 0
        for key in keys1 & keys2:
            bank, preset = key
            list1 = presets1_grouped[key]
            list2 = presets2_grouped[key]

            if len(list1) != len(list2):
                self.errors.append(f"Preset count mismatch for bank={bank}, preset={preset}: {len(list1)} != {len(list2)}")
                continue

            # Compare each preset
            for p1, p2 in zip(list1, list2):
                # Names are outward-facing info, check strictly
                if p1["name"] != p2["name"]:
                    self.errors.append(f"Preset name mismatch for bank={bank}, preset={preset}: \"{p1["name"]}\" != \"{p2["name"]}\"")

                # Other metadata
                if p1["library"] != p2["library"]:
                    self.warnings.append(f"Preset library differs for \"{p1["name"]}\": {p1["library"]} != {p2["library"]}")
                if p1["genre"] != p2["genre"]:
                    self.warnings.append(f"Preset genre differs for \"{p1["name"]}\": {p1["genre"]} != {p2["genre"]}")
                if p1["morphology"] != p2["morphology"]:
                    self.warnings.append(f"Preset morphology differs for \"{p1["name"]}\": {p1["morphology"]} != {p2["morphology"]}")

                # Compare zones
                if len(p1["zones"]) != len(p2["zones"]):
                    self.errors.append(f"Zone count mismatch for preset \"{p1["name"]}\": {len(p1["zones"])} != {len(p2["zones"])}")
                    continue

                for z_idx, (z1, z2) in enumerate(zip(p1["zones"], p2["zones"])):
                    # Compare generators
                    gens1 = sorted([(g["oper"], g["amount"]) for g in z1["generators"]])
                    gens2 = sorted([(g["oper"], g["amount"]) for g in z2["generators"]])

                    # For instrument (oper=41), consider instrument mapping
                    gens1_normalized = []
                    for oper, amount in gens1:
                        if oper == 41:  # instrument
                            # Compare by instrument content (not ID)
                            if 0 <= amount < len(instruments1):
                                inst = instruments1[amount]
                                gens1_normalized.append((41, inst["name"]))  # use name as a stand-in identifier
                            else:
                                gens1_normalized.append((41, amount))
                        else:
                            gens1_normalized.append((oper, amount))

                    gens2_normalized = []
                    for oper, amount in gens2:
                        if oper == 41:  # instrument
                            if 0 <= amount < len(instruments2):
                                inst = instruments2[amount]
                                gens2_normalized.append((41, inst["name"]))
                            else:
                                gens2_normalized.append((41, amount))
                        else:
                            gens2_normalized.append((oper, amount))

                    if gens1_normalized != gens2_normalized:
                        self.errors.append(f"Generator mismatch in preset \"{p1["name"]}\" zone {z_idx}")

                    # Compare modulators
                    mods1 = sorted([(m["src_oper"], m["dest_oper"], m["amount"], m["amt_src_oper"], m["trans_oper"]) for m in z1["modulators"]])
                    mods2 = sorted([(m["src_oper"], m["dest_oper"], m["amount"], m["amt_src_oper"], m["trans_oper"]) for m in z2["modulators"]])

                    if mods1 != mods2:
                        self.errors.append(f"Modulator mismatch in preset \"{p1["name"]}\" zone {z_idx}")

                matched_count += 1

        print(f"  ✓ {matched_count} presets matched and verified")


def main():
    if len(sys.argv) != 3:
        print("Usage: python test_equivalence.py <file1.sf2> <file2.sf2>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    checker = SoundFontEquivalenceChecker(file1, file2)
    result = checker.check()

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
