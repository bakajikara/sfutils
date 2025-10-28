#!/usr/bin/env python3
"""
SF2 Compiler - 展開されたディレクトリ構造からSoundFont2ファイルを再構築するツール

以下の構造からSF2ファイルを生成します:
- bank-info.json: メタデータ
- samples/: オーディオファイル（FLAC/WAV等の波形データ）
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
    print("Error: soundfile library is required.")
    print("Install it with: pip install soundfile")
    sys.exit(1)

from sf2_constants import GENERATOR_IDS


class SF2Compiler:
    """ディレクトリ構造からSF2ファイルを生成するクラス"""

    # サンプル間のパディング量（サンプル数）
    # 最低46サンプル（92バイト）
    SAMPLE_PADDING = 46

    def __init__(self, input_dir, output_sf2):
        self.input_dir = Path(input_dir)
        self.output_sf2 = output_sf2

        # データ格納用
        self.bank_info = {}
        self.samples = []
        self.instruments = []
        self.presets = []

    def compile(self):
        """ディレクトリ構造からSF2ファイルを生成"""
        print(f"Compiling from: {self.input_dir}")

        # 各部分を読み込む
        self._load_bank_info()
        self._load_samples()
        self._load_instruments()
        self._load_presets()

        # SF2ファイルを生成
        print(f"Writing SF2 file: {self.output_sf2}")
        self._write_sf2_file()

        print("Compilation complete!")

    def _load_bank_info(self):
        """bank-info.jsonを読み込む"""
        info_path = self.input_dir / "bank-info.json"

        if not info_path.exists():
            raise FileNotFoundError(f"bank-info.json not found in {self.input_dir}")

        with open(info_path, "r", encoding="utf-8") as f:
            self.bank_info = json.load(f)

        print(f"  Loaded: bank-info.json")

    def _load_samples(self):
        """samplesディレクトリからサンプルメタデータを読み込む（高速化：PCMデータは後で読む）"""
        samples_dir = self.input_dir / "samples"

        if not samples_dir.exists():
            raise FileNotFoundError(f"samples directory not found in {self.input_dir}")

        # JSONファイルを順番に読み込む（アルファベット順）
        json_files = sorted(samples_dir.glob("*.json"))

        for json_path in json_files:
            # JSONからメタデータを読み込む（SF2仕様書のフィールド名）
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # 対応するオーディオファイルのパスを検索（FLAC, WAV, その他）
            audio_path = None
            for ext in [".flac", ".wav", ".ogg", ".aiff", ".aif"]:
                candidate = json_path.with_suffix(ext)
                if candidate.exists():
                    audio_path = candidate
                    break

            if audio_path is None:
                raise FileNotFoundError(f"Audio file not found for: {json_path}")

            # サンプルデータを構築（PCMデータはまだ読まない）
            sample_data = {
                "sample_name": metadata["sample_name"],
                "start": metadata["start"],
                "end": metadata["end"],
                "start_loop": metadata["start_loop"],
                "end_loop": metadata["end_loop"],
                "sample_rate": metadata["sample_rate"],
                "original_key": metadata["original_key"],
                "correction": metadata["correction"],
                "sample_link": metadata["sample_link"],
                "sample_type": metadata["sample_type"],
                "_audio_path": audio_path,  # PCMデータを読むためのパスを保存
            }

            self.samples.append(sample_data)

        print(f"  Loaded: {len(self.samples)} sample files from samples/")

    def _read_pcm_data(self, audio_path):
        """オーディオファイル（FLAC/WAV等）からPCMデータを読み込む（必要な時だけ呼ばれる）"""
        try:
            # soundfileで読み込み（FLAC, WAV, OGG, AIFF等に対応）
            data, samplerate = sf.read(audio_path, dtype="int16")

            # NumPy配列をバイト列に変換
            return data.tobytes()
        except Exception as e:
            raise ValueError(f"Failed to read audio file {audio_path}: {e}")

    def _load_instruments(self):
        """instrumentsディレクトリからJSONファイルを読み込む"""
        instruments_dir = self.input_dir / "instruments"

        if not instruments_dir.exists():
            raise FileNotFoundError(f"instruments directory not found in {self.input_dir}")

        # JSONファイルを順番に読み込む（ファイル名順）
        json_files = sorted(instruments_dir.glob("*.json"))

        for json_path in json_files:
            with open(json_path, "r", encoding="utf-8") as f:
                inst_data = json.load(f)
                self.instruments.append(inst_data)

        print(f"  Loaded: {len(self.instruments)} instrument files from instruments/")

    def _load_presets(self):
        """presetsディレクトリからJSONファイルを読み込む"""
        presets_dir = self.input_dir / "presets"

        if not presets_dir.exists():
            raise FileNotFoundError(f"presets directory not found in {self.input_dir}")

        # JSONファイルを順番に読み込む（ファイル名順）
        json_files = sorted(presets_dir.glob("*.json"))

        for json_path in json_files:
            with open(json_path, "r", encoding="utf-8") as f:
                preset_data = json.load(f)
                self.presets.append(preset_data)

        print(f"  Loaded: {len(self.presets)} preset files from presets/")

    def _write_sf2_file(self):
        """SF2ファイルを書き込む"""
        with open(self.output_sf2, "wb") as f:
            # RIFFヘッダー（サイズは後で更新）
            f.write(b"RIFF")
            riff_size_pos = f.tell()
            f.write(struct.pack("<I", 0))  # プレースホルダー
            f.write(b"sfbk")

            # INFO-list
            info_chunk = self._build_info_chunk()
            f.write(info_chunk)

            # sdta-list (メモリ効率のため直接書き込み)
            self._write_sdta_chunk_direct(f)

            # pdta-list
            pdta_chunk = self._build_pdta_chunk()
            f.write(pdta_chunk)

            # RIFFサイズを更新
            file_size = f.tell()
            f.seek(riff_size_pos)
            f.write(struct.pack("<I", file_size - 8))

    def _build_info_chunk(self):
        """INFO-listチャンクを構築"""
        data = b""

        # バージョン情報（必須・最初）
        version = self.bank_info.get("version", "2.04")
        major, minor = version.split(".")
        ifil_data = struct.pack("<HH", int(major), int(minor))
        data += self._make_chunk(b"ifil", ifil_data)

        # サウンドエンジン（必須・2番目）
        sound_engine = self.bank_info.get("sound_engine", "EMU8000")
        data += self._make_chunk(b"isng", self._make_zstr(sound_engine))

        # バンク名（必須・3番目）
        bank_name = self.bank_info.get("bank_name", "Untitled")
        data += self._make_chunk(b"INAM", self._make_zstr(bank_name))

        # SF2仕様に従った順序で出力（オリジナルファイルと同じ順序）
        # INAM → ICRD → IENG → IPRD → ICOP → ICMT → ISFT

        if "creation_date" in self.bank_info:
            data += self._make_chunk(b"ICRD", self._make_zstr(self.bank_info["creation_date"]))

        if "engineer" in self.bank_info:
            data += self._make_chunk(b"IENG", self._make_zstr(self.bank_info["engineer"]))

        if "product" in self.bank_info:
            data += self._make_chunk(b"IPRD", self._make_zstr(self.bank_info["product"]))

        if "copyright" in self.bank_info:
            data += self._make_chunk(b"ICOP", self._make_zstr(self.bank_info["copyright"]))

        if "comment" in self.bank_info:
            data += self._make_chunk(b"ICMT", self._make_zstr(self.bank_info["comment"]))

        if "software" in self.bank_info:
            data += self._make_chunk(b"ISFT", self._make_zstr(self.bank_info["software"]))

        # LISTチャンクとして包む
        return self._make_list_chunk(b"INFO", data)

    def _make_zstr(self, text):
        """
        RIFF仕様に準拠したゼロ終端文字列を作成
        文字列と終端文字の合計バイト数が偶数になるよう調整
        """
        encoded = text.encode("ascii")
        # 文字列長が奇数の場合、終端文字は1つ（合計で偶数）
        # 文字列長が偶数の場合、終端文字は2つ（合計で偶数）
        if len(encoded) % 2 == 1:
            return encoded + b"\x00"
        else:
            return encoded + b"\x00\x00"

    def _write_sdta_chunk_direct(self, f):
        """sdta-listチャンクを直接ファイルに書き込む（メモリ効率化）"""
        # LISTチャンクヘッダー
        f.write(b"LIST")
        list_size_pos = f.tell()
        f.write(struct.pack("<I", 0))  # プレースホルダー
        f.write(b"sdta")

        # smplチャンクヘッダー
        f.write(b"smpl")
        smpl_size_pos = f.tell()
        f.write(struct.pack("<I", 0))  # プレースホルダー

        smpl_start = f.tell()

        # サンプルデータを順次書き込み（メモリに全て読み込まない）
        padding = b"\x00" * self.SAMPLE_PADDING * 2  # サンプル間のゼロパディング

        for sample in self.samples:
            # PCMデータを必要な時だけ読み込む（高速化）
            pcm = self._read_pcm_data(sample["_audio_path"])
            f.write(pcm)
            f.write(padding)

        # サイズを計算して更新
        smpl_end = f.tell()
        smpl_size = smpl_end - smpl_start

        # smplチャンクサイズを更新
        f.seek(smpl_size_pos)
        f.write(struct.pack("<I", smpl_size))

        # パディング（奇数サイズの場合）
        f.seek(smpl_end)
        if smpl_size % 2:
            f.write(b"\x00")

        # LISTチャンクサイズを更新
        list_end = f.tell()
        list_size = list_end - list_size_pos - 4
        f.seek(list_size_pos)
        f.write(struct.pack("<I", list_size))

        # ファイルポインタを末尾に戻す
        f.seek(list_end)

    def _build_pdta_chunk(self):
        """pdta-list (Hydra) チャンクを構築"""
        # Hydraの各部分を構築
        phdr_data = []
        pbag_data = []
        pmod_data = []
        pgen_data = []

        inst_data = []
        ibag_data = []
        imod_data = []
        igen_data = []

        shdr_data = []

        # サンプルヘッダーを構築（SF2仕様書のフィールド名を使用）
        # 最初のサンプルは0から開始
        current_sample_offset = 0

        for sample in self.samples:
            # sample_name (20バイト)
            name = sample["sample_name"][:20].ljust(20, "\x00").encode("ascii")

            # start, end (絶対位置)
            # デコンパイル時の値をそのまま使用するのではなく、
            # WAVファイルのサイズから再計算
            pcm_size = (sample["end"] - sample["start"]) * 2  # バイト単位のサイズ
            num_frames = pcm_size // 2  # サンプル数

            start = current_sample_offset
            end = start + num_frames

            # start_loop, end_loop（相対位置→絶対位置）
            start_loop = start + (sample["start_loop"] - sample["start"])
            end_loop = start + (sample["end_loop"] - sample["start"])

            shdr_record = struct.pack(
                "<20sIIIIIBBHH",
                name,
                start,
                end,
                start_loop,
                end_loop,
                sample["sample_rate"],
                sample["original_key"],
                sample["correction"] & 0xFF,
                sample["sample_link"],
                sample["sample_type"]
            )

            shdr_data.append(shdr_record)

            # 次のサンプルのオフセット（パディング込み）
            current_sample_offset = end + self.SAMPLE_PADDING  # パディングはサンプル数単位

        # ターミネータ（"EOS"）
        shdr_data.append(b"EOS".ljust(20, b"\x00") + b"\x00" * 26)

        # インストゥルメントを構築
        # サンプル名→インデックスマッピング
        # "0000_SampleName" 形式と "SampleName" 形式の両方をサポート
        sample_name_to_id = {}
        for i, s in enumerate(self.samples):
            # sample_name単体
            sample_name_to_id[s["sample_name"]] = i
            # JSONファイル名から"0000_SampleName"形式も登録
            # JSONファイル名は"0000_SampleName.json"なので、拡張子を除いたものがキー
            # ただし、_audio_pathからファイル名を取得
            json_stem = s["_audio_path"].stem  # "0000_SampleName"
            sample_name_to_id[json_stem] = i

        for inst in self.instruments:
            name = inst["name"][:20].ljust(20, "\x00").encode("ascii")
            bag_ndx = len(ibag_data)

            inst_record = struct.pack("<20sH", name, bag_ndx)
            inst_data.append(inst_record)

            # ゾーンを処理
            for zone in inst["zones"]:
                gen_ndx = len(igen_data)
                mod_ndx = len(imod_data)

                ibag_record = struct.pack("<HH", gen_ndx, mod_ndx)
                ibag_data.append(ibag_record)

                # モジュレータを追加
                if "modulators" in zone:
                    for mod in zone["modulators"]:
                        imod_record = struct.pack(
                            "<HHhHH",
                            mod["src_oper"],
                            mod["dest_oper"],
                            mod["amount"],
                            mod["amt_src_oper"],
                            mod["trans_oper"]
                        )
                        imod_data.append(imod_record)

                # ジェネレータを追加（順序重要）
                generators = zone["generators"]

                # keyRangeとvelRangeを最初に配置
                if "keyRange" in generators:
                    lo, hi = map(int, generators["keyRange"].split("-"))
                    amount = lo | (hi << 8)
                    igen_data.append(struct.pack("<Hh", 43, amount))

                if "velRange" in generators:
                    lo, hi = map(int, generators["velRange"].split("-"))
                    amount = lo | (hi << 8)
                    igen_data.append(struct.pack("<Hh", 44, amount))

                # その他のジェネレータ
                for gen_name, gen_value in generators.items():
                    if gen_name in ["keyRange", "velRange", "sampleID"]:
                        continue

                    gen_id = GENERATOR_IDS.get(gen_name)
                    if gen_id is None:
                        continue

                    igen_data.append(struct.pack("<Hh", gen_id, gen_value))

                # sampleIDを最後に配置
                if "sampleID" in generators:
                    sample_name = generators["sampleID"]
                    sample_id = sample_name_to_id.get(sample_name, 0)
                    igen_data.append(struct.pack("<Hh", 53, sample_id))

        # インストゥルメントのターミネータ
        inst_data.append(b"EOI".ljust(20, b"\x00") + struct.pack("<H", len(ibag_data)))
        ibag_data.append(struct.pack("<HH", len(igen_data), len(imod_data)))

        # モジュレータとジェネレータのターミネータレコードを追加
        imod_data.append(struct.pack("<HHhHH", 0, 0, 0, 0, 0))
        igen_data.append(struct.pack("<Hh", 0, 0))

        # プリセットを構築
        # インストゥルメント名→インデックスマッピング
        inst_name_to_id = {}
        for i, inst in enumerate(self.instruments):
            # インストゥルメント名で検索
            inst_name_to_id[inst["name"]] = i

        for preset in self.presets:
            name = preset["name"][:20].ljust(20, "\x00").encode("ascii")
            preset_num = preset["preset_number"]
            bank = preset["bank"]
            bag_ndx = len(pbag_data)

            # library/genre/morphologyフィールドを読み込む
            library = preset.get("library", 0)
            genre = preset.get("genre", 0)
            morphology = preset.get("morphology", 0)

            phdr_record = struct.pack(
                "<20sHHHIII",
                name,
                preset_num,
                bank,
                bag_ndx,
                library, genre, morphology
            )
            phdr_data.append(phdr_record)

            # ゾーンを処理
            for zone in preset["zones"]:
                gen_ndx = len(pgen_data)
                mod_ndx = len(pmod_data)

                pbag_record = struct.pack("<HH", gen_ndx, mod_ndx)
                pbag_data.append(pbag_record)

                # モジュレータを追加
                if "modulators" in zone:
                    for mod in zone["modulators"]:
                        pmod_record = struct.pack(
                            "<HHhHH",
                            mod["src_oper"],
                            mod["dest_oper"],
                            mod["amount"],
                            mod["amt_src_oper"],
                            mod["trans_oper"]
                        )
                        pmod_data.append(pmod_record)

                # ジェネレータを追加
                generators = zone["generators"]

                # keyRangeとvelRangeを最初に配置
                if "keyRange" in generators:
                    lo, hi = map(int, generators["keyRange"].split("-"))
                    amount = lo | (hi << 8)
                    pgen_data.append(struct.pack("<Hh", 43, amount))

                if "velRange" in generators:
                    lo, hi = map(int, generators["velRange"].split("-"))
                    amount = lo | (hi << 8)
                    pgen_data.append(struct.pack("<Hh", 44, amount))

                # その他のジェネレータ
                for gen_name, gen_value in generators.items():
                    if gen_name in ["keyRange", "velRange", "instrument"]:
                        continue

                    gen_id = GENERATOR_IDS.get(gen_name)
                    if gen_id is None:
                        continue

                    pgen_data.append(struct.pack("<Hh", gen_id, gen_value))

                # instrumentを最後に配置
                if "instrument" in generators:
                    inst_ref = generators["instrument"]
                    inst_id = inst_name_to_id.get(inst_ref, None)
                    if inst_id is None:
                        print(f"Warning: Instrument \"{inst_ref}\" not found in mapping")
                        inst_id = 0
                    pgen_data.append(struct.pack("<Hh", 41, inst_id))

        # プリセットのターミネータ
        phdr_data.append(b"EOP".ljust(20, b"\x00") + struct.pack("<HHHIII", 0, 0, len(pbag_data), 0, 0, 0))
        pbag_data.append(struct.pack("<HH", len(pgen_data), len(pmod_data)))

        # モジュレータとジェネレータのターミネータレコードを追加
        pmod_data.append(struct.pack("<HHhHH", 0, 0, 0, 0, 0))
        pgen_data.append(struct.pack("<Hh", 0, 0))

        # 各チャンクを構築
        result = b""
        result += self._make_chunk(b"phdr", b"".join(phdr_data))
        result += self._make_chunk(b"pbag", b"".join(pbag_data))
        result += self._make_chunk(b"pmod", b"".join(pmod_data))
        result += self._make_chunk(b"pgen", b"".join(pgen_data))
        result += self._make_chunk(b"inst", b"".join(inst_data))
        result += self._make_chunk(b"ibag", b"".join(ibag_data))
        result += self._make_chunk(b"imod", b"".join(imod_data))
        result += self._make_chunk(b"igen", b"".join(igen_data))
        result += self._make_chunk(b"shdr", b"".join(shdr_data))

        # LISTチャンクとして包む
        return self._make_list_chunk(b"pdta", result)

    def _make_chunk(self, chunk_id, data):
        """チャンクを作成"""
        size = len(data)
        result = chunk_id + struct.pack("<I", size) + data

        # パディング
        if size % 2:
            result += b"\x00"

        return result

    def _make_list_chunk(self, list_type, data):
        """LISTチャンクを作成"""
        list_data = list_type + data
        return self._make_chunk(b"LIST", list_data)


def main():
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
