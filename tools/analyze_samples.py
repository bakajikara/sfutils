#!/usr/bin/env python3
"""
サンプル名の重複を詳細に調査するツール
"""

import sys
import struct
from pathlib import Path
from collections import Counter


def get_sample_names(sf2_path):
    """SF2ファイルからサンプル名のリストを取得"""
    samples = []

    with open(sf2_path, "rb") as f:
        # RIFFヘッダーをスキップ
        f.read(12)

        # pdtaチャンクを探す
        while True:
            chunk_id = f.read(4)
            if not chunk_id:
                break
            chunk_size = struct.unpack("<I", f.read(4))[0]
            chunk_data_start = f.tell()

            if chunk_id == b"LIST":
                list_type = f.read(4)
                if list_type == b"pdta":
                    chunk_end = chunk_data_start + chunk_size

                    # shdrチャンクを探す
                    while f.tell() < chunk_end:
                        sub_id = f.read(4)
                        if not sub_id:
                            break
                        sub_size = struct.unpack("<I", f.read(4))[0]
                        sub_data_start = f.tell()

                        if sub_id == b"shdr":
                            print(f"Found \"shdr\" chunk at offset {sub_data_start}, size {sub_size} bytes")
                            # サンプルヘッダーを読む
                            shdr_data = f.read(sub_size)
                            for i in range(0, len(shdr_data), 46):
                                if i + 46 > len(shdr_data):
                                    break
                                name = shdr_data[i:i + 20].decode("ascii", errors="ignore").rstrip("\x00")
                                if name == "EOS":
                                    break
                                samples.append(name)
                            break
                        else:
                            f.seek(sub_data_start + sub_size)
                            if sub_size % 2:
                                f.read(1)
                    # LIST(pdta) を処理したら、そのチャンクの終端に移動してから抜ける
                    f.seek(chunk_data_start + chunk_size)
                    if chunk_size % 2:
                        f.read(1)
                    break

            # 次のチャンクへ（読み進めがあっても常にチャンク開始 + サイズへ移動）
            f.seek(chunk_data_start + chunk_size)
            if chunk_size % 2:
                f.read(1)

    return samples


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <file.sf2>")
        sys.exit(1)

    sf2_file = sys.argv[1]

    if not Path(sf2_file).exists():
        print(f"Error: File not found - {sf2_file}")
        sys.exit(1)

    print(f"Analyzing: {sf2_file}")
    samples = get_sample_names(sf2_file)

    print(f"\nTotal samples: {len(samples)}")

    # 重複を確認
    counter = Counter(samples)
    duplicates = [(name, count) for name, count in counter.items() if count > 1]

    if duplicates:
        print(f"\nDuplicate sample names found: {len(duplicates)}")
        for name, count in sorted(duplicates, key=lambda x: -x[1])[:10]:
            print(f"  \"{name}\": {count} times")
    else:
        print("\nNo duplicate sample names found.")

    # 空の名前をチェック
    empty_names = [i for i, name in enumerate(samples) if not name.strip()]
    if empty_names:
        print(f"\nEmpty sample names at indices: {empty_names[:10]}")
        if len(empty_names) > 10:
            print(f"  ... and {len(empty_names) - 10} more")


if __name__ == "__main__":
    main()
