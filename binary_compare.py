#!/usr/bin/env python3
"""
SF2ファイルをバイナリレベルで詳細比較するツール
各チャンクの内容をバイトレベルで比較し、差異を特定する
"""

import struct
import sys
from pathlib import Path


def parse_sf2_structure(sf2_path):
    """SF2ファイルの構造を解析"""
    with open(sf2_path, "rb") as f:
        data = f.read()

    if data[:4] != b"RIFF":
        raise ValueError("Not a valid RIFF file")

    file_size = struct.unpack("<I", data[4:8])[0]

    if data[8:12] != b"sfbk":
        raise ValueError("Not a valid SF2 file")

    chunks = {}
    pos = 12

    while pos < len(data):
        if pos + 8 > len(data):
            break

        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        chunk_data = data[pos + 8:pos + 8 + chunk_size]

        if chunk_id == b"LIST":
            list_type = chunk_data[:4]
            list_data = chunk_data[4:]

            # LISTチャンクの中身を解析
            if list_type == b"INFO":
                chunks["INFO"] = parse_info_list(list_data)
            elif list_type == b"sdta":
                chunks["sdta"] = parse_sdta_list(list_data)
            elif list_type == b"pdta":
                chunks["pdta"] = parse_pdta_list(list_data)

        pos += 8 + chunk_size
        if chunk_size % 2:
            pos += 1  # パディング

    return chunks


def parse_info_list(data):
    """INFO-listチャンクを解析"""
    info = {}
    pos = 0

    while pos < len(data):
        if pos + 8 > len(data):
            break

        chunk_id = data[pos:pos + 4].decode("ascii", errors="ignore")
        chunk_size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        chunk_data = data[pos + 8:pos + 8 + chunk_size]

        info[chunk_id] = {
            "size": chunk_size,
            "data": chunk_data
        }

        pos += 8 + chunk_size
        if chunk_size % 2:
            pos += 1

    return info


def parse_sdta_list(data):
    """sdta-listチャンクを解析"""
    sdta = {}
    pos = 0

    while pos < len(data):
        if pos + 8 > len(data):
            break

        chunk_id = data[pos:pos + 4].decode("ascii", errors="ignore")
        chunk_size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        chunk_data = data[pos + 8:pos + 8 + chunk_size]

        sdta[chunk_id] = {
            "size": chunk_size,
            "data": chunk_data
        }

        pos += 8 + chunk_size
        if chunk_size % 2:
            pos += 1

    return sdta


def parse_pdta_list(data):
    """pdta-listチャンクを解析"""
    pdta = {}
    pos = 0

    while pos < len(data):
        if pos + 8 > len(data):
            break

        chunk_id = data[pos:pos + 4].decode("ascii", errors="ignore")
        chunk_size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        chunk_data = data[pos + 8:pos + 8 + chunk_size]

        pdta[chunk_id] = {
            "size": chunk_size,
            "data": chunk_data
        }

        pos += 8 + chunk_size
        if chunk_size % 2:
            pos += 1

    return pdta


def compare_bytes(data1, data2, chunk_name):
    """バイトデータを比較して差異を表示"""
    if data1 == data2:
        print(f"  ✓ {chunk_name}: 完全一致 ({len(data1)} bytes)")
        return True

    print(f"  ✗ {chunk_name}: 差異あり")
    print(f"    Original: {len(data1)} bytes")
    print(f"    Rebuilt:  {len(data2)} bytes")
    print(f"    Diff:     {len(data2) - len(data1):+d} bytes")

    # 最初の差異を見つける
    min_len = min(len(data1), len(data2))
    first_diff = None

    for i in range(min_len):
        if data1[i] != data2[i]:
            first_diff = i
            break

    if first_diff is not None:
        print(f"    First difference at byte {first_diff}:")

        # 前後のコンテキストを表示
        start = max(0, first_diff - 8)
        end = min(min_len, first_diff + 8)

        print(f"      Original: {data1[start:end].hex(" ")}")
        print(f"      Rebuilt:  {data2[start:end].hex(" ")}")

        # ASCIIとして表示可能な場合
        try:
            orig_str = data1[start:end].decode("ascii", errors="ignore")
            rebu_str = data2[start:end].decode("ascii", errors="ignore")
            if orig_str.isprintable() or rebu_str.isprintable():
                print(f"      Original (ASCII): \"{orig_str}\"")
                print(f"      Rebuilt  (ASCII): \"{rebu_str}\"")
        except:
            pass

    if len(data1) != len(data2):
        if len(data1) > len(data2):
            print(f"    Original has {len(data1) - len(data2)} extra bytes at end")
        else:
            print(f"    Rebuilt has {len(data2) - len(data1)} extra bytes at end")

    return False


def compare_info_chunks(info1, info2):
    """INFOチャンクを比較"""
    print("\nINFO-list comparison:")

    all_keys = set(info1.keys()) | set(info2.keys())

    for key in sorted(all_keys):
        if key not in info1:
            print(f"  ✗ {key}: Missing in original")
        elif key not in info2:
            print(f"  ✗ {key}: Missing in rebuilt")
        else:
            compare_bytes(info1[key]["data"], info2[key]["data"], key)


def compare_sdta_chunks(sdta1, sdta2):
    """sdtaチャンクを比較"""
    print("\nsdta-list comparison:")

    all_keys = set(sdta1.keys()) | set(sdta2.keys())

    for key in sorted(all_keys):
        if key not in sdta1:
            print(f"  ✗ {key}: Missing in original")
        elif key not in sdta2:
            print(f"  ✗ {key}: Missing in rebuilt")
        else:
            # smplチャンクは大きいので最初と最後だけ比較
            if key == "smpl":
                data1 = sdta1[key]["data"]
                data2 = sdta2[key]["data"]

                if data1 == data2:
                    print(f"  ✓ {key}: 完全一致 ({len(data1):,} bytes)")
                else:
                    print(f"  ✗ {key}: 差異あり ({len(data1):,} vs {len(data2):,} bytes)")

                    # 最初の1000バイトを比較
                    if data1[:1000] == data2[:1000]:
                        print(f"    First 1000 bytes: 一致")
                    else:
                        print(f"    First 1000 bytes: 差異あり")
                        for i in range(min(1000, len(data1), len(data2))):
                            if data1[i] != data2[i]:
                                print(f"      First diff at byte {i}")
                                print(f"        Original: {data1[max(0, i - 8):i + 8].hex(" ")}")
                                print(f"        Rebuilt:  {data2[max(0, i - 8):i + 8].hex(" ")}")
                                break

                    # 最後の1000バイトを比較
                    if data1[-1000:] == data2[-1000:]:
                        print(f"    Last 1000 bytes: 一致")
                    else:
                        print(f"    Last 1000 bytes: 差異あり")
            else:
                compare_bytes(sdta1[key]["data"], sdta2[key]["data"], key)


def compare_pdta_chunks(pdta1, pdta2):
    """pdtaチャンクを比較"""
    print("\npdta-list comparison:")

    chunk_order = ["phdr", "pbag", "pmod", "pgen", "inst", "ibag", "imod", "igen", "shdr"]

    for key in chunk_order:
        if key not in pdta1:
            print(f"  ✗ {key}: Missing in original")
        elif key not in pdta2:
            print(f"  ✗ {key}: Missing in rebuilt")
        else:
            compare_bytes(pdta1[key]["data"], pdta2[key]["data"], key)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <original.sf2> <rebuilt.sf2>")
        sys.exit(1)

    original_path = sys.argv[1]
    rebuilt_path = sys.argv[2]

    print(f"Comparing:")
    print(f"  Original: {original_path}")
    print(f"  Rebuilt:  {rebuilt_path}")
    print("=" * 70)

    # ファイル全体のサイズを比較
    orig_size = Path(original_path).stat().st_size
    rebu_size = Path(rebuilt_path).stat().st_size

    print(f"\nFile sizes:")
    print(f"  Original: {orig_size:,} bytes")
    print(f"  Rebuilt:  {rebu_size:,} bytes")
    print(f"  Diff:     {rebu_size - orig_size:+,} bytes")

    if orig_size == rebu_size:
        print("  ✓ File sizes match")
    else:
        print("  ✗ File sizes differ")

    # 構造を解析
    print("\nParsing files...")
    chunks1 = parse_sf2_structure(original_path)
    chunks2 = parse_sf2_structure(rebuilt_path)

    # チャンクごとに比較
    if "INFO" in chunks1 and "INFO" in chunks2:
        compare_info_chunks(chunks1["INFO"], chunks2["INFO"])

    if "sdta" in chunks1 and "sdta" in chunks2:
        compare_sdta_chunks(chunks1["sdta"], chunks2["sdta"])

    if "pdta" in chunks1 and "pdta" in chunks2:
        compare_pdta_chunks(chunks1["pdta"], chunks2["pdta"])

    print("\n" + "=" * 70)
    print("Comparison complete")


if __name__ == "__main__":
    main()
