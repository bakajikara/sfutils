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
        """samplesディレクトリからサンプルメタデータを読み込む（高速化:PCMデータは後で読む）"""
        samples_dir = self.input_dir / "samples"

        if not samples_dir.exists():
            raise FileNotFoundError(f"samples directory not found in {self.input_dir}")

        # JSONファイルを読み込む
        json_files = samples_dir.glob("*.json")

        for json_path in json_files:
            # JSONからメタデータを読み込む
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

            sample_type = metadata.get("sample_type", "mono")

            if sample_type == "stereo":
                # ステレオサンプル: 左右に分離して2つのサンプルとして登録
                # 左右で開始位置やループは同じ
                start = metadata.get("start", 0)
                end = metadata.get("end", 0)
                start_loop = metadata.get("start_loop", 0)
                end_loop = metadata.get("end_loop", 0)
                original_key = metadata.get("original_key", 60)
                correction = metadata.get("correction", 0)

                # 左チャンネル用サンプル
                left_sample = {
                    "sample_name": metadata["sample_name"],
                    "start": start,
                    "end": end,
                    "start_loop": start_loop,
                    "end_loop": end_loop,
                    "original_key": original_key,
                    "correction": correction,
                    "sample_link": None,  # 後で設定
                    "sample_type": 4,  # left
                    "_audio_path": audio_path,
                    "_channel": "left",
                    "_is_stereo": True
                }

                # 右チャンネル用サンプル
                right_sample = {
                    "sample_name": metadata["sample_name"],
                    "start": start,
                    "end": end,
                    "start_loop": start_loop,
                    "end_loop": end_loop,
                    "original_key": original_key,
                    "correction": correction,
                    "sample_link": None,  # 後で設定
                    "sample_type": 2,  # right
                    "_audio_path": audio_path,
                    "_channel": "right",
                    "_is_stereo": True
                }

                # 相互リンクを設定
                left_idx = len(self.samples)
                right_idx = left_idx + 1
                left_sample["sample_link"] = right_idx
                right_sample["sample_link"] = left_idx

                self.samples.append(left_sample)
                self.samples.append(right_sample)

            else:
                # モノラルサンプル
                sample_data = {
                    "sample_name": metadata["sample_name"],
                    "start": metadata.get("start", 0),
                    "end": metadata.get("end", 0),
                    "start_loop": metadata.get("start_loop", 0),
                    "end_loop": metadata.get("end_loop", 0),
                    "original_key": metadata.get("original_key", 60),
                    "correction": metadata.get("correction", 0),
                    "sample_link": 0,
                    "sample_type": 1,  # mono
                    "_audio_path": audio_path,
                    "_channel": None,
                    "_is_stereo": False
                }

                self.samples.append(sample_data)

        print(f"  Loaded: {len(self.samples)} sample entries from samples/")

    def _read_pcm_data(self, audio_path, channel=None):
        """オーディオファイル（FLAC/WAV等）からPCMデータを読み込む（必要な時だけ呼ばれる）

        Args:
            audio_path: オーディオファイルのパス
            channel: "left", "right", またはNone(モノラル)
        """
        try:
            # soundfileで読み込み（FLAC, WAV, OGG, AIFF等に対応）
            data, samplerate = sf.read(audio_path, dtype="int16")

            # チャンネル分離
            if channel == "left":
                if len(data.shape) == 2:  # ステレオ
                    data = data[:, 0]
                # モノラルの場合はそのまま
            elif channel == "right":
                if len(data.shape) == 2:  # ステレオ
                    data = data[:, 1]
                # モノラルの場合はそのまま
            else:
                # モノラル、またはステレオをそのまま返す
                pass

            # NumPy配列をバイト列に変換
            return data.tobytes(), samplerate
        except Exception as e:
            raise ValueError(f"Failed to read audio file {audio_path}: {e}")

    def _load_instruments(self):
        """instrumentsディレクトリからJSONファイルを読み込む"""
        instruments_dir = self.input_dir / "instruments"

        if not instruments_dir.exists():
            raise FileNotFoundError(f"instruments directory not found in {self.input_dir}")

        # JSONファイルを読み込む
        json_files = instruments_dir.glob("*.json")

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

        # JSONファイルを読み込む
        json_files = presets_dir.glob("*.json")

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
        # 各サンプルの絶対位置を記録
        padding = b"\x00" * self.SAMPLE_PADDING * 2  # サンプル間のゼロパディング
        current_offset = 0  # サンプル単位でのオフセット

        for sample in self.samples:
            # PCMデータを必要な時だけ読み込み、チャンネル情報を渡す
            channel = sample.get("_channel")
            pcm, sample_rate = self._read_pcm_data(sample["_audio_path"], channel)

            # サンプル数を計算（int16なので2バイトで1サンプル）
            num_samples = len(pcm) // 2

            # 絶対位置を計算（相対位置 + 現在のオフセット）
            sample["_absolute_start"] = current_offset + sample["start"]
            sample["_absolute_end"] = current_offset + sample["end"]
            sample["_absolute_start_loop"] = current_offset + sample["start_loop"]
            sample["_absolute_end_loop"] = current_offset + sample["end_loop"]
            sample["_sample_rate"] = sample_rate

            # データを書き込む
            f.write(pcm)
            f.write(padding)

            # 次のサンプルのオフセットを更新
            current_offset += num_samples + self.SAMPLE_PADDING

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

        # サンプルヘッダーを構築
        for idx, sample in enumerate(self.samples):
            # sample_nameを動的生成（左右のサフィックスを追加）
            base_name = sample["sample_name"]
            if sample.get("_is_stereo"):
                channel = sample.get("_channel")
                if channel == "left":
                    name = f"{base_name}_L"[:19]
                else:  # right
                    name = f"{base_name}_R"[:19]
            else:
                name = base_name[:19]

            name_bytes = name.ljust(20, "\x00").encode("ascii")

            # 絶対位置を使用（_write_sdta_chunk_directで計算済み）
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

        # ターミネータ（"EOS"）
        shdr_data.append(b"EOS".ljust(20, b"\x00") + b"\x00" * 26)

        # インストゥルメントを構築
        # サンプル名+チャンネル→インデックスマッピング
        sample_name_to_id = {}
        for i, s in enumerate(self.samples):
            base_name = s["sample_name"]

            # ベース名だけでもアクセス可能に（モノラルの場合）
            if not s.get("_is_stereo"):
                sample_name_to_id[base_name] = i

            # チャンネル情報付きでもアクセス可能に（ステレオの場合）
            channel = s.get("_channel")
            if channel:
                sample_name_to_id[(base_name, channel)] = i

        for inst in self.instruments:
            name = inst["name"][:19].ljust(20, "\x00").encode("ascii")
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
                    if gen_name in ["keyRange", "velRange", "sample"]:
                        continue

                    gen_id = GENERATOR_IDS.get(gen_name)
                    if gen_id is None:
                        continue

                    igen_data.append(struct.pack("<Hh", gen_id, gen_value))

                # sample (旧sampleID)を最後に配置
                if "sample" in generators:
                    sample_name = generators["sample"]
                    sample_channel = generators.get("sample_channel")  # "left", "right", またはNone

                    # サンプルIDを検索
                    if sample_channel:
                        # ステレオサンプルの場合、チャンネル情報を使用
                        sample_id = sample_name_to_id.get((sample_name, sample_channel), 0)
                    else:
                        # モノラルサンプルの場合
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
            name = preset["name"][:19].ljust(20, "\x00").encode("ascii")
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
