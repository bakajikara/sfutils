#!/usr/bin/env python3
"""
SF2 Decompiler - SoundFont2ファイルを展開するツール

SF2ファイルを以下の構造に展開します:
- bank-info.json: メタデータ
- samples/: FLACファイル（波形データ）
- instruments/: インストゥルメント定義（JSON）
- presets/: プリセット定義（JSON）
"""

import os
import sys
import json
import struct
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    print("Error: soundfile library is required. Install it with: pip install soundfile")
    sys.exit(1)

import numpy as np

from sf2_constants import GENERATOR_NAMES


class SF2Parser:
    """SF2ファイルのパーサー"""

    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
        self.info_data = {}
        self.sample_data = b""
        self.sample_data_24 = b""
        self.pdta = {}

    def parse(self):
        """SF2ファイル全体を解析"""
        with open(self.filepath, "rb") as f:
            self.file = f

            # RIFFヘッダー
            riff_id = f.read(4)
            if riff_id != b"RIFF":
                raise ValueError("Not a RIFF file")

            file_size = struct.unpack("<I", f.read(4))[0]

            form_type = f.read(4)
            if form_type != b"sfbk":
                raise ValueError("Not a SoundFont file")

            # 3つのメインチャンクを解析
            self._parse_info_chunk()
            self._parse_sdta_chunk()
            self._parse_pdta_chunk()

    def _read_chunk_header(self):
        """チャンクヘッダーを読む"""
        chunk_id = self.file.read(4)
        chunk_size = struct.unpack("<I", self.file.read(4))[0]
        return chunk_id, chunk_size

    def _parse_info_chunk(self):
        """INFO-listチャンクを解析"""
        chunk_id, chunk_size = self._read_chunk_header()
        if chunk_id != b"LIST":
            raise ValueError("Expected LIST chunk")

        list_type = self.file.read(4)
        if list_type != b"INFO":
            raise ValueError("Expected INFO list")

        chunk_end = self.file.tell() + chunk_size - 4

        while self.file.tell() < chunk_end:
            sub_id, sub_size = self._read_chunk_header()
            data = self.file.read(sub_size)

            # パディング処理
            if sub_size % 2:
                self.file.read(1)

            # データを格納
            sub_id_str = sub_id.decode("ascii", errors="ignore")

            if sub_id == b"ifil":
                # バージョン情報
                major, minor = struct.unpack("<HH", data)
                self.info_data["version"] = f"{major}.{minor:02d}"
            elif sub_id == b"isng":
                self.info_data["sound_engine"] = data.decode("ascii", errors="ignore").rstrip("\x00")
            elif sub_id == b"INAM":
                self.info_data["bank_name"] = data.decode("ascii", errors="ignore").rstrip("\x00")
            elif sub_id == b"ICOP":
                self.info_data["copyright"] = data.decode("ascii", errors="ignore").rstrip("\x00")
            elif sub_id == b"ICMT":
                self.info_data["comment"] = data.decode("ascii", errors="ignore").rstrip("\x00")
            elif sub_id == b"ISFT":
                self.info_data["software"] = data.decode("ascii", errors="ignore").rstrip("\x00")
            elif sub_id == b"ICRD":
                self.info_data["creation_date"] = data.decode("ascii", errors="ignore").rstrip("\x00")
            elif sub_id == b"IENG":
                self.info_data["engineer"] = data.decode("ascii", errors="ignore").rstrip("\x00")
            elif sub_id == b"IPRD":
                self.info_data["product"] = data.decode("ascii", errors="ignore").rstrip("\x00")
            else:
                # その他のINFOチャンク
                self.info_data[sub_id_str.lower()] = data.decode("ascii", errors="ignore").rstrip("\x00")

    def _parse_sdta_chunk(self):
        """sdta-listチャンクを解析"""
        chunk_id, chunk_size = self._read_chunk_header()
        if chunk_id != b"LIST":
            raise ValueError("Expected LIST chunk")

        list_type = self.file.read(4)
        if list_type != b"sdta":
            raise ValueError("Expected sdta list")

        chunk_end = self.file.tell() + chunk_size - 4

        while self.file.tell() < chunk_end:
            sub_id, sub_size = self._read_chunk_header()

            if sub_id == b"smpl":
                # 16-bit サンプルデータ
                self.sample_data = self.file.read(sub_size)
            elif sub_id == b"sm24":
                # 24-bit LSB (未実装でOK)
                self.sample_data_24 = self.file.read(sub_size)
            else:
                # 不明なチャンクはスキップ
                self.file.read(sub_size)

            # パディング処理
            if sub_size % 2:
                self.file.read(1)

    def _parse_pdta_chunk(self):
        """pdta-list (Hydra) チャンクを解析"""
        chunk_id, chunk_size = self._read_chunk_header()
        if chunk_id != b"LIST":
            raise ValueError("Expected LIST chunk")

        list_type = self.file.read(4)
        if list_type != b"pdta":
            raise ValueError("Expected pdta list")

        chunk_end = self.file.tell() + chunk_size - 4

        # 9つのサブチャンクを順番に読み込む
        hydra_chunks = ["phdr", "pbag", "pmod", "pgen", "inst", "ibag", "imod", "igen", "shdr"]

        for expected_chunk in hydra_chunks:
            if self.file.tell() >= chunk_end:
                break

            sub_id, sub_size = self._read_chunk_header()
            sub_id_str = sub_id.decode("ascii", errors="ignore")

            data = self.file.read(sub_size)

            # パディング処理
            if sub_size % 2:
                self.file.read(1)

            # データを格納
            self.pdta[sub_id_str] = data

    def get_preset_headers(self):
        """プリセットヘッダーを取得"""
        data = self.pdta.get("phdr", b"")
        headers = []

        # sfPresetHeader = 38 bytes
        for i in range(0, len(data), 38):
            if i + 38 > len(data):
                break

            chunk = data[i:i + 38]
            name = chunk[0:20].decode("ascii", errors="ignore").rstrip("\x00")
            preset, bank, bag_ndx = struct.unpack("<HHH", chunk[20:26])

            # ターミネータレコードはスキップ
            if name == "EOP":
                break

            headers.append({
                "name": name,
                "preset": preset,
                "bank": bank,
                "bag_ndx": bag_ndx,
                "library": struct.unpack("<I", chunk[26:30])[0],
                "genre": struct.unpack("<I", chunk[30:34])[0],
                "morphology": struct.unpack("<I", chunk[34:38])[0]
            })

        return headers

    def get_preset_bags(self):
        """プリセットバッグ（ゾーン）を取得"""
        data = self.pdta.get("pbag", b"")
        bags = []

        # sfPresetBag = 4 bytes
        for i in range(0, len(data), 4):
            if i + 4 > len(data):
                break

            gen_ndx, mod_ndx = struct.unpack("<HH", data[i:i + 4])
            bags.append({
                "gen_ndx": gen_ndx,
                "mod_ndx": mod_ndx
            })

        return bags

    def get_preset_generators(self):
        """プリセットジェネレータを取得"""
        data = self.pdta.get("pgen", b"")
        generators = []

        # sfGenList = 4 bytes
        for i in range(0, len(data), 4):
            if i + 4 > len(data):
                break

            oper, amount = struct.unpack("<Hh", data[i:i + 4])
            generators.append({
                "oper": oper,
                "amount": amount
            })

        return generators

    def get_preset_modulators(self):
        """プリセットモジュレータを取得"""
        data = self.pdta.get("pmod", b"")
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

    def get_instrument_headers(self):
        """インストゥルメントヘッダーを取得"""
        data = self.pdta.get("inst", b"")
        headers = []

        # sfInst = 22 bytes
        for i in range(0, len(data), 22):
            if i + 22 > len(data):
                break

            name = data[i:i + 20].decode("ascii", errors="ignore").rstrip("\x00")
            bag_ndx = struct.unpack("<H", data[i + 20:i + 22])[0]

            # ターミネータレコードはスキップ
            if name == "EOI":
                break

            headers.append({
                "name": name,
                "bag_ndx": bag_ndx
            })

        return headers

    def get_instrument_bags(self):
        """インストゥルメントバッグ（ゾーン）を取得"""
        data = self.pdta.get("ibag", b"")
        bags = []

        # sfInstBag = 4 bytes
        for i in range(0, len(data), 4):
            if i + 4 > len(data):
                break

            gen_ndx, mod_ndx = struct.unpack("<HH", data[i:i + 4])
            bags.append({
                "gen_ndx": gen_ndx,
                "mod_ndx": mod_ndx
            })

        return bags

    def get_instrument_generators(self):
        """インストゥルメントジェネレータを取得"""
        data = self.pdta.get("igen", b"")
        generators = []

        # sfGenList = 4 bytes
        for i in range(0, len(data), 4):
            if i + 4 > len(data):
                break

            oper, amount = struct.unpack("<Hh", data[i:i + 4])
            generators.append({
                "oper": oper,
                "amount": amount
            })

        return generators

    def get_instrument_modulators(self):
        """インストゥルメントモジュレータを取得"""
        data = self.pdta.get("imod", b"")
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

    def get_sample_headers(self):
        """サンプルヘッダーを取得"""
        data = self.pdta.get("shdr", b"")
        headers = []

        # sfSample = 46 bytes
        for i in range(0, len(data), 46):
            if i + 46 > len(data):
                break

            chunk = data[i:i + 46]
            name = chunk[0:20].decode("ascii", errors="ignore").rstrip("\x00")

            # ターミネータレコードはスキップ
            if name == "EOS":
                break

            values = struct.unpack("<IIIIIBbHH", chunk[20:46])

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


class SF2Decompiler:
    """SF2ファイルをディレクトリ構造に展開するクラス"""

    def __init__(self, sf2_path, output_dir):
        self.sf2_path = sf2_path
        self.output_dir = Path(output_dir)
        self.parser = SF2Parser(sf2_path)

    def decompile(self):
        """SF2ファイルを展開"""
        print(f"Parsing SF2 file: {self.sf2_path}")
        self.parser.parse()

        # 出力ディレクトリを作成
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Decompiling to: {self.output_dir}")

        # 各部分を展開
        self._export_bank_info()
        self._export_samples()
        self._export_instruments()
        self._export_presets()

        print("Decompilation complete!")

    def _export_bank_info(self):
        """bank-info.jsonを出力"""
        output_path = self.output_dir / "bank-info.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.parser.info_data, f, indent=2, ensure_ascii=False)

        print(f"  Created: bank-info.json")

    def _export_samples(self):
        """samplesディレクトリにFLACファイルとメタデータを出力"""
        samples_dir = self.output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        sample_headers = self.parser.get_sample_headers()
        sample_data = self.parser.sample_data

        # サンプルデータを16bit PCMとして解釈
        # 各サンプルは2バイト(int16)

        for idx, header in enumerate(sample_headers):
            name = header["name"]
            start = header["start"]  # サンプル単位
            end = header["end"]

            # サンプルデータを抽出（バイト単位に変換）
            pcm_bytes = sample_data[start * 2:end * 2]

            # int16配列に変換
            pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)

            # 元の順序を保持するため、ファイル名の先頭にインデックスを追加
            sanitized_name = self._sanitize_filename(name)
            base_filename = f"{idx:04d}_{sanitized_name}"

            flac_path = samples_dir / f"{base_filename}.flac"
            json_path = samples_dir / f"{base_filename}.json"

            # FLACファイルを書き込む（soundfileを使用）
            sf.write(flac_path, pcm_array, header["sample_rate"], subtype="PCM_16")

            # メタデータをJSONで保存（SF2仕様書のフィールド名を使用）
            metadata = {
                "sample_name": name,
                "start": header["start"],
                "end": header["end"],
                "start_loop": header["start_loop"],
                "end_loop": header["end_loop"],
                "sample_rate": header["sample_rate"],
                "original_key": header["original_key"],
                "correction": header["correction"],
                "sample_link": header["sample_link"],
                "sample_type": header["sample_type"]
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"  Created: {len(sample_headers)} sample files in samples/")

    def _export_instruments(self):
        """instrumentsディレクトリにJSONファイルを出力"""
        instruments_dir = self.output_dir / "instruments"
        instruments_dir.mkdir(exist_ok=True)

        inst_headers = self.parser.get_instrument_headers()
        inst_bags = self.parser.get_instrument_bags()
        inst_gens = self.parser.get_instrument_generators()
        inst_mods = self.parser.get_instrument_modulators()
        sample_headers = self.parser.get_sample_headers()

        for idx, inst in enumerate(inst_headers):
            # このインストゥルメントのゾーン範囲を取得
            bag_start = inst["bag_ndx"]
            bag_end = inst_headers[idx + 1]["bag_ndx"] if idx + 1 < len(inst_headers) else len(inst_bags) - 1

            zones = []

            for bag_idx in range(bag_start, bag_end):
                if bag_idx >= len(inst_bags):
                    break

                bag = inst_bags[bag_idx]

                # ジェネレータとモジュレータの範囲を取得
                gen_start = bag["gen_ndx"]
                gen_end = inst_bags[bag_idx + 1]["gen_ndx"] if bag_idx + 1 < len(inst_bags) else len(inst_gens)

                mod_start = bag["mod_ndx"]
                mod_end = inst_bags[bag_idx + 1]["mod_ndx"] if bag_idx + 1 < len(inst_bags) else len(inst_mods)

                # ジェネレータを解析
                generators = {}
                for gen_idx in range(gen_start, gen_end):
                    if gen_idx >= len(inst_gens):
                        break
                    gen = inst_gens[gen_idx]
                    gen_name = GENERATOR_NAMES.get(gen["oper"], f"unknown_{gen["oper"]}")

                    # 特殊処理
                    if gen["oper"] == 43:  # keyRange
                        lo = gen["amount"] & 0xFF
                        hi = (gen["amount"] >> 8) & 0xFF
                        generators[gen_name] = f"{lo}-{hi}"
                    elif gen["oper"] == 44:  # velRange
                        lo = gen["amount"] & 0xFF
                        hi = (gen["amount"] >> 8) & 0xFF
                        generators[gen_name] = f"{lo}-{hi}"
                    elif gen["oper"] == 53:  # sampleID
                        if gen["amount"] < len(sample_headers):
                            # インデックス_サンプル名の形式で保存（重複名対策）
                            sample_name = sample_headers[gen["amount"]]["name"]
                            generators[gen_name] = f"{gen["amount"]:04d}_{sample_name}"
                        else:
                            generators[gen_name] = gen["amount"]
                    else:
                        generators[gen_name] = gen["amount"]

                # モジュレータを解析
                modulators = []
                for mod_idx in range(mod_start, mod_end):
                    if mod_idx >= len(inst_mods):
                        break
                    modulators.append(inst_mods[mod_idx])

                zone = {}

                zone["generators"] = generators

                if modulators:
                    zone["modulators"] = modulators

                zones.append(zone)

            # JSONとして保存
            inst_data = {
                "name": inst["name"],
                "zones": zones
            }

            # 元の順序を保持するため、ファイル名の先頭にインデックスを追加
            filename = f"{idx:04d}_{self._sanitize_filename(inst["name"])}.json"
            output_path = instruments_dir / filename

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(inst_data, f, indent=2, ensure_ascii=False)

        print(f"  Created: {len(inst_headers)} instrument files in instruments/")

    def _export_presets(self):
        """presetsディレクトリにJSONファイルを出力"""
        presets_dir = self.output_dir / "presets"
        presets_dir.mkdir(exist_ok=True)

        preset_headers = self.parser.get_preset_headers()
        preset_bags = self.parser.get_preset_bags()
        preset_gens = self.parser.get_preset_generators()
        preset_mods = self.parser.get_preset_modulators()
        inst_headers = self.parser.get_instrument_headers()

        for idx, preset in enumerate(preset_headers):
            # このプリセットのゾーン範囲を取得
            bag_start = preset["bag_ndx"]
            bag_end = preset_headers[idx + 1]["bag_ndx"] if idx + 1 < len(preset_headers) else len(preset_bags) - 1

            zones = []

            for bag_idx in range(bag_start, bag_end):
                if bag_idx >= len(preset_bags):
                    break

                bag = preset_bags[bag_idx]

                # ジェネレータとモジュレータの範囲を取得
                gen_start = bag["gen_ndx"]
                gen_end = preset_bags[bag_idx + 1]["gen_ndx"] if bag_idx + 1 < len(preset_bags) else len(preset_gens)

                mod_start = bag["mod_ndx"]
                mod_end = preset_bags[bag_idx + 1]["mod_ndx"] if bag_idx + 1 < len(preset_bags) else len(preset_mods)

                # ジェネレータを解析
                generators = {}
                for gen_idx in range(gen_start, gen_end):
                    if gen_idx >= len(preset_gens):
                        break
                    gen = preset_gens[gen_idx]
                    gen_name = GENERATOR_NAMES.get(gen["oper"], f"unknown_{gen["oper"]}")

                    # 特殊処理
                    if gen["oper"] == 43:  # keyRange
                        lo = gen["amount"] & 0xFF
                        hi = (gen["amount"] >> 8) & 0xFF
                        generators[gen_name] = f"{lo}-{hi}"
                    elif gen["oper"] == 44:  # velRange
                        lo = gen["amount"] & 0xFF
                        hi = (gen["amount"] >> 8) & 0xFF
                        generators[gen_name] = f"{lo}-{hi}"
                    elif gen["oper"] == 41:  # instrument
                        if gen["amount"] < len(inst_headers):
                            # インストゥルメント名を使用
                            inst_name = inst_headers[gen["amount"]]["name"]
                            generators[gen_name] = inst_name
                        else:
                            generators[gen_name] = gen["amount"]
                    else:
                        generators[gen_name] = gen["amount"]

                # モジュレータを解析
                modulators = []
                for mod_idx in range(mod_start, mod_end):
                    if mod_idx >= len(preset_mods):
                        break
                    modulators.append(preset_mods[mod_idx])

                # グローバルゾーンかどうか判定
                is_global = "instrument" not in generators

                zone = {
                    "generators": generators
                }

                # グローバルゾーンの場合のみis_globalを追加
                if is_global:
                    zone["is_global"] = True

                if modulators:
                    zone["modulators"] = modulators

                zones.append(zone)

            # JSONとして保存
            preset_data = {
                "name": preset["name"],
                "bank": preset["bank"],
                "preset_number": preset["preset"],
                "library": preset["library"],
                "genre": preset["genre"],
                "morphology": preset["morphology"],
                "zones": zones
            }

            # 元の順序を保持するため、ファイル名の先頭にインデックスを追加
            filename = f"{idx:04d}_{preset["bank"]:03d}-{preset["preset"]:03d}_{self._sanitize_filename(preset["name"])}.json"
            output_path = presets_dir / filename

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(preset_data, f, indent=2, ensure_ascii=False)

        print(f"  Created: {len(preset_headers)} preset files in presets/")

    def _sanitize_filename(self, name):
        """ファイル名に使用できない文字を置換"""
        # Windows/Linuxで使用できない文字を置換
        invalid_chars = "<>:\"/\\|?*"
        for char in invalid_chars:
            name = name.replace(char, "_")
        return name.strip()


def main():
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
