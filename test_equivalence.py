#!/usr/bin/env python3
"""
SF2等価性テスト - 2つのSF2ファイルが実用上等価であることを検証

このテストは以下を無視します:
- サンプル、インストゥルメント、プリセットの順序
- 内部的なサンプル名などの名称の微差
- 順序の違いによるID/offset値の差異

以下は検出します:
- 外向き（ユーザーに見える）のプリセット名などの違い
- 参照関係の不整合（順序が違っても参照先が正しいか確認）
- オーディオデータの違い
- ジェネレータやモジュレータの設定値の違い
"""

import sys
import struct
import numpy as np
from pathlib import Path
from collections import defaultdict


class SF2EquivalenceChecker:
    """2つのSF2ファイルの実用的な等価性をチェック"""

    def __init__(self, file1_path, file2_path):
        self.file1_path = Path(file1_path)
        self.file2_path = Path(file2_path)
        self.errors = []
        self.warnings = []

    def check(self):
        """等価性チェックを実行"""
        print("=" * 80)
        print("SF2 Equivalence Test")
        print("=" * 80)
        print(f"File 1: {self.file1_path}")
        print(f"File 2: {self.file2_path}")
        print()

        # 両ファイルをパース
        print("Parsing files...")
        data1 = self._parse_sf2(self.file1_path)
        data2 = self._parse_sf2(self.file2_path)

        if data1 is None or data2 is None:
            print("❌ Failed to parse one or both files")
            return False

        print(f"  File 1: {len(data1["samples"])} samples, {len(data1["instruments"])} instruments, {len(data1["presets"])} presets")
        print(f"  File 2: {len(data2["samples"])} samples, {len(data2["instruments"])} instruments, {len(data2["presets"])} presets")
        print()

        # 各要素をチェック
        print("Checking equivalence...")
        print()

        self._check_info(data1["info"], data2["info"])
        self._check_samples(data1["samples"], data2["samples"], data1["sample_data"], data2["sample_data"])
        self._check_instruments(data1["instruments"], data2["instruments"], data1["samples"], data2["samples"])
        self._check_presets(data1["presets"], data2["presets"], data1["instruments"], data2["instruments"])

        # 結果表示
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
            print("   (順序や内部名称の違いを除く)")
            return True

    def _parse_sf2(self, filepath):
        """SF2ファイルをパース"""
        try:
            with open(filepath, "rb") as f:
                # RIFFヘッダー
                riff_id = f.read(4)
                if riff_id != b"RIFF":
                    self.errors.append(f"{filepath}: Not a RIFF file")
                    return None

                file_size = struct.unpack("<I", f.read(4))[0]
                form_type = f.read(4)
                if form_type != b"sfbk":
                    self.errors.append(f"{filepath}: Not a SoundFont file")
                    return None

                # データ構造
                data = {
                    "info": {},
                    "sample_data": b"",
                    "samples": [],
                    "instruments": [],
                    "presets": [],
                    "pdta_raw": {}
                }

                # 3つのメインチャンクを読む
                for _ in range(3):
                    chunk_id = f.read(4)
                    if len(chunk_id) < 4:
                        break
                    chunk_size = struct.unpack("<I", f.read(4))[0]
                    list_type = f.read(4)
                    chunk_end = f.tell() + chunk_size - 4

                    if list_type == b"INFO":
                        data["info"] = self._parse_info_chunk(f, chunk_end)
                    elif list_type == b"sdta":
                        data["sample_data"] = self._parse_sdta_chunk(f, chunk_end)
                    elif list_type == b"pdta":
                        data["pdta_raw"] = self._parse_pdta_chunk(f, chunk_end)

                    f.seek(chunk_end)

                # pdtaデータを構造化
                self._structure_pdta(data)

                return data

        except Exception as e:
            self.errors.append(f"Failed to parse {filepath}: {e}")
            return None

    def _parse_info_chunk(self, f, chunk_end):
        """INFOチャンクをパース"""
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
        """sdtaチャンクをパース（サンプルデータ）"""
        sample_data = b""
        while f.tell() < chunk_end:
            sub_id = f.read(4)
            if len(sub_id) < 4:
                break
            sub_size = struct.unpack("<I", f.read(4))[0]

            if sub_id == b"smpl":
                sample_data = f.read(sub_size)
            else:
                f.read(sub_size)

            if sub_size % 2:
                f.read(1)

        return sample_data

    def _parse_pdta_chunk(self, f, chunk_end):
        """pdtaチャンクをパース"""
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
        """pdtaの生データを構造化"""
        pdta = data["pdta_raw"]

        # サンプルヘッダー
        shdr_data = pdta.get("shdr", b"")
        for i in range(0, len(shdr_data), 46):
            if i + 46 > len(shdr_data):
                break
            chunk = shdr_data[i:i + 46]
            name = chunk[0:20].decode("ascii", errors="ignore").rstrip("\x00")
            if name == "EOS":
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

        # インストゥルメントヘッダーとバッグ
        inst_data = pdta.get("inst", b"")
        ibag_data = pdta.get("ibag", b"")
        igen_data = pdta.get("igen", b"")
        imod_data = pdta.get("imod", b"")

        inst_headers = []
        for i in range(0, len(inst_data), 22):
            if i + 22 > len(inst_data):
                break
            name = inst_data[i:i + 20].decode("ascii", errors="ignore").rstrip("\x00")
            if name == "EOI":
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

        # インストゥルメント構造化
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

        # プリセットヘッダーとバッグ
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
            if name == "EOP":
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

        # プリセット構造化
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
        """INFOチャンクの比較（外向き情報のみ）"""
        print("Checking INFO (metadata)...")

        # 外向き情報（ユーザーに見える）
        external_keys = ["bank_name", "version"]

        for key in external_keys:
            val1 = info1.get(key, "")
            val2 = info2.get(key, "")
            if val1 != val2:
                self.errors.append(f"INFO.{key}: \"{val1}\" != \"{val2}\"")

        # その他の情報は警告のみ
        internal_keys = ["sound_engine", "copyright"]
        for key in internal_keys:
            val1 = info1.get(key, "")
            val2 = info2.get(key, "")
            if val1 != val2:
                self.warnings.append(f"INFO.{key}: \"{val1}\" != \"{val2}\" (内部情報の違い)")

        print("  ✓ INFO check complete")

    def _check_samples(self, samples1, samples2, data1, data2):
        """サンプルの比較（順序を無視）"""
        print("Checking samples...")

        if len(samples1) != len(samples2):
            self.errors.append(f"Sample count mismatch: {len(samples1)} != {len(samples2)}")

        # サンプルをオーディオデータでマッピング（順序を無視）
        def extract_sample_data(sample, data):
            """サンプルのPCMデータを抽出"""
            start = sample["start"] * 2  # 16-bit = 2 bytes
            end = sample["end"] * 2
            if end > len(data):
                end = len(data)
            return data[start:end]

        # ファイル1のサンプルをPCMデータでインデックス化
        import hashlib
        samples1_by_audio = {}
        samples1_hashes = {}  # idx -> hash（後で使う）

        for idx, sample in enumerate(samples1):
            pcm = extract_sample_data(sample, data1)
            # ハッシュを使う（メモリ効率）
            pcm_hash = hashlib.md5(pcm).hexdigest()
            samples1_by_audio[pcm_hash] = (idx, sample, pcm)
            samples1_hashes[idx] = pcm_hash

        # ファイル2の各サンプルをマッチング
        matched = set()
        sample_mapping = {}  # file2_idx -> file1_idx
        samples2_hashes = {}  # idx -> hash（後で使う）

        for idx2, sample2 in enumerate(samples2):
            pcm2 = extract_sample_data(sample2, data2)
            pcm2_hash = hashlib.md5(pcm2).hexdigest()
            samples2_hashes[idx2] = pcm2_hash

            if pcm2_hash in samples1_by_audio:
                idx1, sample1, pcm1 = samples1_by_audio[pcm2_hash]
                matched.add(idx1)
                sample_mapping[idx2] = idx1

                # メタデータ比較（名前以外）
                if sample1["sample_rate"] != sample2["sample_rate"]:
                    self.errors.append(f"Sample #{idx2}: sample_rate mismatch")
                if sample1["original_key"] != sample2["original_key"]:
                    self.errors.append(f"Sample #{idx2}: original_key mismatch")
                if sample1["correction"] != sample2["correction"]:
                    self.errors.append(f"Sample #{idx2}: correction mismatch")
                if sample1["sample_type"] != sample2["sample_type"]:
                    self.errors.append(f"Sample #{idx2}: sample_type mismatch")

                # ループポイントの比較（オーディオデータ内での相対位置）
                loop1_start_rel = sample1["start_loop"] - sample1["start"]
                loop1_end_rel = sample1["end_loop"] - sample1["start"]
                loop2_start_rel = sample2["start_loop"] - sample2["start"]
                loop2_end_rel = sample2["end_loop"] - sample2["start"]

                if loop1_start_rel != loop2_start_rel or loop1_end_rel != loop2_end_rel:
                    self.errors.append(f"Sample #{idx2}: loop points mismatch")

            else:
                self.errors.append(f"Sample #{idx2} in file2 not found in file1")

        # 未マッチのサンプル確認
        for idx1 in range(len(samples1)):
            if idx1 not in matched:
                self.errors.append(f"Sample #{idx1} in file1 not found in file2")

        print(f"  ✓ {len(matched)}/{len(samples1)} samples matched")

        # マッピングとハッシュを保存（後で使う）
        self.sample_mapping = sample_mapping
        self.samples1_hashes = samples1_hashes
        self.samples2_hashes = samples2_hashes

    def _check_instruments(self, instruments1, instruments2, samples1, samples2):
        """インストゥルメントの比較（順序を無視）"""
        print("Checking instruments...")

        if len(instruments1) != len(instruments2):
            self.errors.append(f"Instrument count mismatch: {len(instruments1)} != {len(instruments2)}")

        # インストゥルメントを内容でマッピング
        def inst_signature(inst, sample_hashes):
            """インストゥルメントの署名を生成（名前を除く）"""
            sig = []
            for zone in inst["zones"]:
                zone_gens = []
                zone_mods = []

                for gen in zone["generators"]:
                    # sampleID (oper=53) の場合、サンプルの実際のPCMハッシュを使用
                    if gen["oper"] == 53:  # sampleID
                        sample_idx = gen["amount"]
                        if sample_idx in sample_hashes:
                            # サンプルのハッシュ値を使って一意に識別
                            zone_gens.append((53, sample_hashes[sample_idx]))
                        else:
                            zone_gens.append((53, f"invalid_{sample_idx}"))
                    else:
                        zone_gens.append((gen["oper"], gen["amount"]))

                for mod in zone["modulators"]:
                    zone_mods.append((mod["src_oper"], mod["dest_oper"], mod["amount"], mod["amt_src_oper"], mod["trans_oper"]))

                # ジェネレータとモジュレータを別々にソートしてから結合
                sig.append((tuple(sorted(zone_gens)), tuple(sorted(zone_mods))))

            return tuple(sorted(sig))

        # ファイル1のインストゥルメントをインデックス化
        inst1_by_sig = {}
        for idx, inst in enumerate(instruments1):
            sig = inst_signature(inst, self.samples1_hashes)
            if sig not in inst1_by_sig:
                inst1_by_sig[sig] = []
            inst1_by_sig[sig].append((idx, inst))

        # ファイル2の各インストゥルメントをマッチング
        matched = set()
        inst_mapping = {}  # file2_idx -> file1_idx

        for idx2, inst2 in enumerate(instruments2):
            sig2 = inst_signature(inst2, self.samples2_hashes)

            if sig2 in inst1_by_sig and inst1_by_sig[sig2]:
                idx1, inst1 = inst1_by_sig[sig2].pop(0)
                matched.add(idx1)
                inst_mapping[idx2] = idx1

                # 名前の違いは警告のみ（内部名称）
                if inst1["name"] != inst2["name"]:
                    self.warnings.append(f"Instrument name differs: \"{inst1["name"]}\" vs \"{inst2["name"]}\" (内部名称)")

            else:
                self.errors.append(f"Instrument #{idx2} \"{inst2["name"]}\" in file2 not found in file1")

        # 未マッチのインストゥルメント確認
        for sig, remaining in inst1_by_sig.items():
            for idx1, inst1 in remaining:
                if idx1 not in matched:
                    self.errors.append(f"Instrument #{idx1} \"{inst1["name"]}\" in file1 not found in file2")

        print(f"  ✓ {len(matched)}/{len(instruments1)} instruments matched")

        # マッピングを保存
        self.inst_mapping = inst_mapping

    def _check_presets(self, presets1, presets2, instruments1, instruments2):
        """プリセットの比較（順序を無視）"""
        print("Checking presets...")

        if len(presets1) != len(presets2):
            self.errors.append(f"Preset count mismatch: {len(presets1)} != {len(presets2)}")

        # プリセットをbank/preset番号でグループ化
        def group_by_bank_preset(presets):
            """bank/preset番号でグループ化"""
            grouped = defaultdict(list)
            for preset in presets:
                key = (preset["bank"], preset["preset"])
                grouped[key].append(preset)
            return grouped

        presets1_grouped = group_by_bank_preset(presets1)
        presets2_grouped = group_by_bank_preset(presets2)

        # bank/preset番号の一致を確認
        keys1 = set(presets1_grouped.keys())
        keys2 = set(presets2_grouped.keys())

        if keys1 != keys2:
            missing_in_2 = keys1 - keys2
            missing_in_1 = keys2 - keys1
            if missing_in_2:
                self.errors.append(f"Preset bank/preset numbers in file1 but not in file2: {missing_in_2}")
            if missing_in_1:
                self.errors.append(f"Preset bank/preset numbers in file2 but not in file1: {missing_in_1}")

        # 各bank/preset番号内でプリセットを比較
        matched_count = 0
        for key in keys1 & keys2:
            bank, preset = key
            list1 = presets1_grouped[key]
            list2 = presets2_grouped[key]

            if len(list1) != len(list2):
                self.errors.append(f"Preset count mismatch for bank={bank}, preset={preset}: {len(list1)} != {len(list2)}")
                continue

            # 各プリセットを比較
            for p1, p2 in zip(list1, list2):
                # 名前は外向き情報なので厳密にチェック
                if p1["name"] != p2["name"]:
                    self.errors.append(f"Preset name mismatch for bank={bank}, preset={preset}: \"{p1["name"]}\" != \"{p2["name"]}\"")

                # その他のメタデータ
                if p1["library"] != p2["library"]:
                    self.warnings.append(f"Preset library differs for \"{p1["name"]}\": {p1["library"]} != {p2["library"]}")
                if p1["genre"] != p2["genre"]:
                    self.warnings.append(f"Preset genre differs for \"{p1["name"]}\": {p1["genre"]} != {p2["genre"]}")
                if p1["morphology"] != p2["morphology"]:
                    self.warnings.append(f"Preset morphology differs for \"{p1["name"]}\": {p1["morphology"]} != {p2["morphology"]}")

                # ゾーンの比較
                if len(p1["zones"]) != len(p2["zones"]):
                    self.errors.append(f"Zone count mismatch for preset \"{p1["name"]}\": {len(p1["zones"])} != {len(p2["zones"])}")
                    continue

                for z_idx, (z1, z2) in enumerate(zip(p1["zones"], p2["zones"])):
                    # ジェネレータの比較
                    gens1 = sorted([(g["oper"], g["amount"]) for g in z1["generators"]])
                    gens2 = sorted([(g["oper"], g["amount"]) for g in z2["generators"]])

                    # instrument (oper=41) の場合、インストゥルメントマッピングを考慮
                    gens1_normalized = []
                    for oper, amount in gens1:
                        if oper == 41:  # instrument
                            # インストゥルメントの内容で比較（IDではなく）
                            if 0 <= amount < len(instruments1):
                                inst = instruments1[amount]
                                gens1_normalized.append((41, inst["name"]))  # 仮に名前で識別
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

                    # モジュレータの比較
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

    checker = SF2EquivalenceChecker(file1, file2)
    result = checker.check()

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
