#!/usr/bin/env python3
"""
SF2ファイル比較ツール - 2つのSF2ファイルの違いを詳細に分析
"""

import sys
import struct
from pathlib import Path


def read_chunk_header(f):
    """チャンクヘッダーを読む"""
    chunk_id = f.read(4)
    if len(chunk_id) < 4:
        return None, 0
    chunk_size = struct.unpack("<I", f.read(4))[0]
    return chunk_id, chunk_size


def analyze_sf2_structure(filepath):
    """SF2ファイルの構造を解析"""
    structure = {}

    with open(filepath, "rb") as f:
        # RIFFヘッダー
        riff_id = f.read(4)
        file_size = struct.unpack("<I", f.read(4))[0]
        form_type = f.read(4)

        structure["file_size"] = file_size + 8
        structure["chunks"] = {}

        # 3つのメインLISTを解析
        for _ in range(3):
            chunk_id, chunk_size = read_chunk_header(f)
            if chunk_id != b"LIST":
                break

            list_type = f.read(4)
            list_type_str = list_type.decode("ascii", errors="ignore")
            chunk_end = f.tell() + chunk_size - 4

            if list_type == b"INFO":
                # INFOチャンクの詳細
                info_chunks = {}
                while f.tell() < chunk_end:
                    sub_id, sub_size = read_chunk_header(f)
                    if not sub_id:
                        break
                    info_chunks[sub_id.decode("ascii", errors="ignore")] = sub_size
                    f.read(sub_size)
                    if sub_size % 2:
                        f.read(1)
                structure["chunks"]["INFO"] = {
                    "total_size": chunk_size + 8,
                    "sub_chunks": info_chunks
                }

            elif list_type == b"sdta":
                # sdtaチャンクの詳細
                sdta_chunks = {}
                while f.tell() < chunk_end:
                    sub_id, sub_size = read_chunk_header(f)
                    if not sub_id:
                        break
                    sdta_chunks[sub_id.decode("ascii", errors="ignore")] = sub_size
                    f.seek(f.tell() + sub_size)
                    if sub_size % 2:
                        f.read(1)
                structure["chunks"]["sdta"] = {
                    "total_size": chunk_size + 8,
                    "sub_chunks": sdta_chunks
                }

            elif list_type == b"pdta":
                # pdtaチャンクの詳細
                pdta_chunks = {}
                while f.tell() < chunk_end:
                    sub_id, sub_size = read_chunk_header(f)
                    if not sub_id:
                        break
                    sub_id_str = sub_id.decode("ascii", errors="ignore")
                    pdta_chunks[sub_id_str] = sub_size
                    f.seek(f.tell() + sub_size)
                    if sub_size % 2:
                        f.read(1)
                structure["chunks"]["pdta"] = {
                    "total_size": chunk_size + 8,
                    "sub_chunks": pdta_chunks
                }

            # パディング処理
            f.seek(chunk_end)

    return structure


def compare_structures(file1, file2):
    """2つのSF2ファイルの構造を比較"""
    print(f"Analyzing: {file1}")
    struct1 = analyze_sf2_structure(file1)

    print(f"Analyzing: {file2}")
    struct2 = analyze_sf2_structure(file2)

    print("\n" + "=" * 70)
    print("FILE SIZE COMPARISON")
    print("=" * 70)
    print(f"Original: {struct1["file_size"]:,} bytes")
    print(f"Rebuilt:  {struct2["file_size"]:,} bytes")
    print(f"Diff:     {struct2["file_size"] - struct1["file_size"]:+,} bytes")

    # 各チャンクの比較
    for chunk_name in ["INFO", "sdta", "pdta"]:
        if chunk_name in struct1["chunks"] and chunk_name in struct2["chunks"]:
            print(f"\n{chunk_name}-list chunk:")
            size1 = struct1["chunks"][chunk_name]["total_size"]
            size2 = struct2["chunks"][chunk_name]["total_size"]
            diff = size2 - size1
            print(f"  Original: {size1:,} bytes")
            print(f"  Rebuilt:  {size2:,} bytes")
            print(f"  Diff:     {diff:+,} bytes")

            # サブチャンクの比較
            sub1 = struct1["chunks"][chunk_name].get("sub_chunks", {})
            sub2 = struct2["chunks"][chunk_name].get("sub_chunks", {})

            all_sub_keys = set(sub1.keys()) | set(sub2.keys())
            for sub_key in sorted(all_sub_keys):
                s1 = sub1.get(sub_key, 0)
                s2 = sub2.get(sub_key, 0)
                if s1 != s2:
                    print(f"    {sub_key}: {s1:,} -> {s2:,} ({s2 - s1:+,} bytes)")

    # pdtaの詳細分析
    if "pdta" in struct1["chunks"] and "pdta" in struct2["chunks"]:
        print("\n" + "=" * 70)
        print("PDTA (Hydra) DETAILED COMPARISON")
        print("=" * 70)

        sub1 = struct1["chunks"]["pdta"]["sub_chunks"]
        sub2 = struct2["chunks"]["pdta"]["sub_chunks"]

        hydra_info = {
            "phdr": ("Preset Headers", 38),
            "pbag": ("Preset Bags", 4),
            "pmod": ("Preset Modulators", 10),
            "pgen": ("Preset Generators", 4),
            "inst": ("Instrument Headers", 22),
            "ibag": ("Instrument Bags", 4),
            "imod": ("Instrument Modulators", 10),
            "igen": ("Instrument Generators", 4),
            "shdr": ("Sample Headers", 46)
        }

        for key, (name, record_size) in hydra_info.items():
            if key in sub1 and key in sub2:
                count1 = sub1[key] // record_size
                count2 = sub2[key] // record_size
                print(f"{key} ({name}):")
                print(f"  Original: {count1} records ({sub1[key]:,} bytes)")
                print(f"  Rebuilt:  {count2} records ({sub2[key]:,} bytes)")
                if count1 != count2:
                    print(f"  *** MISMATCH: {count2 - count1:+d} records ***")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <original.sf2> <rebuilt.sf2>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    if not Path(file1).exists():
        print(f"Error: File not found - {file1}")
        sys.exit(1)

    if not Path(file2).exists():
        print(f"Error: File not found - {file2}")
        sys.exit(1)

    compare_structures(file1, file2)


if __name__ == "__main__":
    main()
