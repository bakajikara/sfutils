#!/usr/bin/env python3
"""
特定の差異を詳しく調査するツール
"""

from sf2_constants import GENERATOR_NAMES
import struct
import sys
from pathlib import Path

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_pdta_list(data):
    """pdta LISTチャンクを解析"""
    chunks = {}
    pos = 0

    while pos < len(data):
        if pos + 8 > len(data):
            break

        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        chunk_data = data[pos + 8:pos + 8 + chunk_size]

        chunks[chunk_id.decode("latin1")] = chunk_data

        pos += 8 + chunk_size
        if chunk_size % 2:
            pos += 1

    return chunks


def analyze_phdr_diff(orig_phdr, rebuilt_phdr, offset=64):
    """phdrの差異を分析"""
    print(f"\n=== phdr差異分析 (offset={offset}) ===")

    # sfPresetHeader構造: 20バイト + 18バイト = 38バイト
    preset_size = 38
    preset_idx = offset // preset_size
    local_offset = offset % preset_size

    print(f"プリセット番号: {preset_idx}")
    print(f"フィールド内オフセット: {local_offset}")

    # プリセットヘッダーのフィールド
    # char achPresetName[20] - offset 0
    # WORD wPreset - offset 20
    # WORD wBank - offset 22
    # WORD wPresetBagNdx - offset 24
    # DWORD dwLibrary - offset 26
    # DWORD dwGenre - offset 30
    # DWORD dwMorphology - offset 34

    preset_start = preset_idx * preset_size
    orig_preset = orig_phdr[preset_start:preset_start + preset_size]
    rebuilt_preset = rebuilt_phdr[preset_start:preset_start + preset_size]

    print(f"\nオリジナル:")
    print_phdr_record(orig_preset)
    print(f"\nリビルド:")
    print_phdr_record(rebuilt_preset)

    if local_offset < 20:
        print(f"\n差異: プリセット名")
    elif local_offset < 22:
        print(f"\n差異: プリセット番号")
    elif local_offset < 24:
        print(f"\n差異: バンク番号")
    elif local_offset < 26:
        print(f"\n差異: プリセットバッグインデックス")
    elif local_offset < 30:
        print(f"\n差異: ライブラリ")
    elif local_offset < 34:
        print(f"\n差異: ジャンル")
    else:
        print(f"\n差異: モーフォロジー")


def print_phdr_record(data):
    """phdrレコードを表示"""
    name = data[:20].rstrip(b"\x00").decode("latin1", errors="replace")
    preset = struct.unpack("<H", data[20:22])[0]
    bank = struct.unpack("<H", data[22:24])[0]
    bag_idx = struct.unpack("<H", data[24:26])[0]
    library = struct.unpack("<I", data[26:30])[0]
    genre = struct.unpack("<I", data[30:34])[0]
    morphology = struct.unpack("<I", data[34:38])[0]

    print(f"  Name: {name}")
    print(f"  Preset: {preset}, Bank: {bank}")
    print(f"  BagIndex: {bag_idx}")
    print(f"  Library: {library}, Genre: {genre}, Morphology: {morphology}")


def analyze_pgen_diff(orig_pgen, rebuilt_pgen, offset=8486):
    """pgenの差異を分析"""
    print(f"\n=== pgen差異分析 (offset={offset}) ===")

    # sfGenList構造: 4バイト
    gen_idx = offset // 4

    print(f"ジェネレータ番号: {gen_idx}")

    # 前後数個を表示
    start_idx = max(0, gen_idx - 5)
    end_idx = min(len(orig_pgen) // 4, gen_idx + 6)

    print(f"\nオリジナル (#{start_idx}〜#{end_idx - 1}):")
    for i in range(start_idx, end_idx):
        gen_data = orig_pgen[i * 4:(i + 1) * 4]
        oper = struct.unpack("<H", gen_data[0:2])[0]
        amount = struct.unpack("<h", gen_data[2:4])[0]
        marker = " <--" if i == gen_idx else ""
        print(f"  #{i}: oper={oper}, amount={amount}{marker}")

    print(f"\nリビルド (#{start_idx}〜#{end_idx - 1}):")
    for i in range(start_idx, end_idx):
        gen_data = rebuilt_pgen[i * 4:(i + 1) * 4]
        oper = struct.unpack("<H", gen_data[0:2])[0]
        amount = struct.unpack("<h", gen_data[2:4])[0]
        marker = " <--" if i == gen_idx else ""
        print(f"  #{i}: oper={oper}, amount={amount}{marker}")


def analyze_igen_diff(orig_igen, rebuilt_igen, offset=4818):
    """igenの差異を分析"""
    print(f"\n=== igen差異分析 (offset={offset}) ===")

    # sfInstGenList構造: 4バイト
    gen_idx = offset // 4

    print(f"ジェネレータ番号: {gen_idx}")

    # 前後数個を表示
    start_idx = max(0, gen_idx - 5)
    end_idx = min(len(orig_igen) // 4, gen_idx + 6)

    print(f"\nオリジナル (#{start_idx}〜#{end_idx - 1}):")
    for i in range(start_idx, end_idx):
        gen_data = orig_igen[i * 4:(i + 1) * 4]
        oper = struct.unpack("<H", gen_data[0:2])[0]
        amount = struct.unpack("<h", gen_data[2:4])[0]
        marker = " <--" if i == gen_idx else ""
        oper_name = get_generator_name(oper)
        print(f"  #{i}: {oper_name}({oper}) = {amount}{marker}")

    print(f"\nリビルド (#{start_idx}〜#{end_idx - 1}):")
    for i in range(start_idx, end_idx):
        gen_data = rebuilt_igen[i * 4:(i + 1) * 4]
        oper = struct.unpack("<H", gen_data[0:2])[0]
        amount = struct.unpack("<h", gen_data[2:4])[0]
        marker = " <--" if i == gen_idx else ""
        oper_name = get_generator_name(oper)
        print(f"  #{i}: {oper_name}({oper}) = {amount}{marker}")


def get_generator_name(oper):
    """ジェネレータオペレータ番号から名前を取得"""
    return GENERATOR_NAMES.get(oper, f"unknown({oper})")


def analyze_shdr_diff(orig_shdr, rebuilt_shdr, offset=66):
    """shdrの差異を分析"""
    print(f"\n=== shdr差異分析 (offset={offset}) ===")

    # sfSample構造: 46バイト
    sample_size = 46
    sample_idx = offset // sample_size
    local_offset = offset % sample_size

    print(f"サンプル番号: {sample_idx}")
    print(f"フィールド内オフセット: {local_offset}")

    # サンプルヘッダーのフィールド
    # char achSampleName[20] - offset 0
    # DWORD dwStart - offset 20
    # DWORD dwEnd - offset 24
    # DWORD dwStartloop - offset 28
    # DWORD dwEndloop - offset 32
    # DWORD dwSampleRate - offset 36
    # BYTE byOriginalPitch - offset 40
    # CHAR chPitchCorrection - offset 41
    # WORD wSampleLink - offset 42
    # WORD sfSampleType - offset 44

    sample_start = sample_idx * sample_size
    orig_sample = orig_shdr[sample_start:sample_start + sample_size]
    rebuilt_sample = rebuilt_shdr[sample_start:sample_start + sample_size]

    print(f"\nオリジナル:")
    print_shdr_record(orig_sample)
    print(f"\nリビルド:")
    print_shdr_record(rebuilt_sample)

    if local_offset < 20:
        print(f"\n差異: サンプル名")
    elif local_offset < 24:
        print(f"\n差異: 開始位置")
    elif local_offset < 28:
        print(f"\n差異: 終了位置")
    elif local_offset < 32:
        print(f"\n差異: ループ開始")
    elif local_offset < 36:
        print(f"\n差異: ループ終了")
    elif local_offset < 40:
        print(f"\n差異: サンプルレート")
    elif local_offset < 41:
        print(f"\n差異: オリジナルピッチ")
    elif local_offset < 42:
        print(f"\n差異: ピッチ補正")
    elif local_offset < 44:
        print(f"\n差異: サンプルリンク")
    else:
        print(f"\n差異: サンプルタイプ")


def print_shdr_record(data):
    """shdrレコードを表示"""
    name = data[:20].rstrip(b"\x00").decode("latin1", errors="replace")
    start = struct.unpack("<I", data[20:24])[0]
    end = struct.unpack("<I", data[24:28])[0]
    startloop = struct.unpack("<I", data[28:32])[0]
    endloop = struct.unpack("<I", data[32:36])[0]
    sample_rate = struct.unpack("<I", data[36:40])[0]
    orig_pitch = data[40]
    pitch_corr = struct.unpack("<b", data[41:42])[0]
    sample_link = struct.unpack("<H", data[42:44])[0]
    sample_type = struct.unpack("<H", data[44:46])[0]

    print(f"  Name: {name}")
    print(f"  Start: {start}, End: {end}")
    print(f"  Loop: {startloop} - {endloop}")
    print(f"  SampleRate: {sample_rate}")
    print(f"  Pitch: {orig_pitch}, Correction: {pitch_corr}")
    print(f"  Link: {sample_link}, Type: {sample_type}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python debug_diff.py <original.sf2> <rebuilt.sf2>")
        sys.exit(1)

    orig_path = Path(sys.argv[1])
    rebuilt_path = Path(sys.argv[2])

    # SF2ファイルを読み込んでpdtaチャンクを抽出
    with open(orig_path, "rb") as f:
        orig_data = f.read()

    with open(rebuilt_path, "rb") as f:
        rebuilt_data = f.read()

    # pdta LISTチャンクを見つける
    orig_pdta = extract_pdta(orig_data)
    rebuilt_pdta = extract_pdta(rebuilt_data)

    orig_chunks = parse_pdta_list(orig_pdta)
    rebuilt_chunks = parse_pdta_list(rebuilt_pdta)

    # 各差異を分析
    analyze_phdr_diff(orig_chunks["phdr"], rebuilt_chunks["phdr"], 64)
    analyze_pgen_diff(orig_chunks["pgen"], rebuilt_chunks["pgen"], 8486)
    analyze_igen_diff(orig_chunks["igen"], rebuilt_chunks["igen"], 4818)
    analyze_shdr_diff(orig_chunks["shdr"], rebuilt_chunks["shdr"], 66)


def extract_pdta(data):
    """SF2データからpdta LISTチャンクのデータ部分を抽出"""
    pos = 12  # RIFF header skip

    while pos < len(data):
        if pos + 8 > len(data):
            break

        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack("<I", data[pos + 4:pos + 8])[0]

        if chunk_id == b"LIST":
            list_type = data[pos + 8:pos + 12]
            if list_type == b"pdta":
                return data[pos + 12:pos + 8 + chunk_size]

        pos += 8 + chunk_size
        if chunk_size % 2:
            pos += 1

    return None


if __name__ == "__main__":
    main()
