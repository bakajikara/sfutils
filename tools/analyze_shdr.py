#!/usr/bin/env python3
"""
Sample headers全体を比較して、正確なパディングを特定する
"""

import struct
import sys


def parse_shdr(sf2_path):
    """shdrチャンクを解析してサンプルヘッダーを表示"""
    with open(sf2_path, "rb") as f:
        data = f.read()

    # pdta LISTチャンクを見つける
    pos = 12
    while pos < len(data):
        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack("<I", data[pos + 4:pos + 8])[0]

        if chunk_id == b"LIST":
            list_type = data[pos + 8:pos + 12]
            if list_type == b"pdta":
                # shdrチャンクを見つける
                pdta_data = data[pos + 12:pos + 8 + chunk_size]
                pdta_pos = 0

                while pdta_pos < len(pdta_data):
                    sub_id = pdta_data[pdta_pos:pdta_pos + 4]
                    sub_size = struct.unpack("<I", pdta_data[pdta_pos + 4:pdta_pos + 8])[0]

                    if sub_id == b"shdr":
                        shdr_data = pdta_data[pdta_pos + 8:pdta_pos + 8 + sub_size]

                        # 最初の3つのサンプルヘッダーを表示
                        for i in range(min(3, sub_size // 46)):
                            header = shdr_data[i * 46:(i + 1) * 46]
                            name = header[:20].rstrip(b"\x00").decode("latin1", errors="replace")
                            start = struct.unpack("<I", header[20:24])[0]
                            end = struct.unpack("<I", header[24:28])[0]
                            startloop = struct.unpack("<I", header[28:32])[0]
                            endloop = struct.unpack("<I", header[32:36])[0]

                            print(f"Sample #{i}: {name}")
                            print(f"  Start: {start}, End: {end} (Length: {end - start})")
                            print(f"  Loop: {startloop} - {endloop}")

                            if i > 0:
                                # 前のサンプルとの間隔を計算
                                prev_header = shdr_data[(i - 1) * 46:i * 46]
                                prev_end = struct.unpack("<I", prev_header[24:28])[0]
                                gap = start - prev_end
                                print(f"  Gap from previous: {gap} samples ({gap * 2} bytes)")
                            print()

                        return

                    pdta_pos += 8 + sub_size
                    if sub_size % 2:
                        pdta_pos += 1
                return

        pos += 8 + chunk_size
        if chunk_size % 2:
            pos += 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_shdr.py <file.sf2>")
        sys.exit(1)

    parse_shdr(sys.argv[1])
