#!/usr/bin/env python3
"""
モジュレータとジェネレータの詳細比較ツール
どのゾーンでレコード数が異なるかを特定する
"""

import struct
import sys
from pathlib import Path


def parse_sf2_pdta(sf2_path):
    """SF2ファイルからpdtaチャンクを解析"""
    with open(sf2_path, "rb") as f:
        data = f.read()

    # pdtaチャンクを探す
    pdta_pos = data.find(b"LIST")
    while pdta_pos >= 0:
        list_size = struct.unpack("<I", data[pdta_pos + 4:pdta_pos + 8])[0]
        list_type = data[pdta_pos + 8:pdta_pos + 12]

        if list_type == b"pdta":
            break

        pdta_pos = data.find(b"LIST", pdta_pos + 1)

    if pdta_pos < 0:
        raise ValueError("pdta-list not found")

    pdta_start = pdta_pos + 12
    pdta_end = pdta_pos + 8 + list_size
    pdta_data = data[pdta_start:pdta_end]

    # 各サブチャンクを解析
    result = {}
    pos = 0

    while pos < len(pdta_data):
        chunk_id = pdta_data[pos:pos + 4]
        chunk_size = struct.unpack("<I", pdta_data[pos + 4:pos + 8])[0]
        chunk_data = pdta_data[pos + 8:pos + 8 + chunk_size]

        result[chunk_id.decode("ascii")] = chunk_data

        pos += 8 + chunk_size
        if chunk_size % 2:
            pos += 1  # パディング

    return result


def analyze_preset_zones(pdta):
    """プリセットゾーンのモジュレータ・ジェネレータを詳細分析"""
    phdr_data = pdta["phdr"]
    pbag_data = pdta["pbag"]
    pmod_data = pdta["pmod"]
    pgen_data = pdta["pgen"]

    num_presets = len(phdr_data) // 38 - 1  # 最後はターミネータ
    num_bags = len(pbag_data) // 4 - 1
    num_mods = len(pmod_data) // 10
    num_gens = len(pgen_data) // 4

    print(f"Preset Analysis:")
    print(f"  Presets: {num_presets}")
    print(f"  Bags: {num_bags}")
    print(f"  Modulators: {num_mods}")
    print(f"  Generators: {num_gens}")
    print()

    # 各プリセットのゾーンを確認
    preset_issues = []

    for i in range(num_presets):
        offset = i * 38
        name = phdr_data[offset:offset + 20].rstrip(b"\x00").decode("ascii", errors="ignore")
        preset_num = struct.unpack("<H", phdr_data[offset + 20:offset + 22])[0]
        bank = struct.unpack("<H", phdr_data[offset + 22:offset + 24])[0]
        bag_start = struct.unpack("<H", phdr_data[offset + 24:offset + 26])[0]

        # 次のプリセットのbag_startを取得
        next_offset = (i + 1) * 38
        bag_end = struct.unpack("<H", phdr_data[next_offset + 24:next_offset + 26])[0]

        num_zones = bag_end - bag_start

        # 各ゾーンのモジュレータとジェネレータ数を確認
        for z in range(num_zones):
            bag_idx = bag_start + z
            bag_offset = bag_idx * 4

            gen_start = struct.unpack("<H", pbag_data[bag_offset:bag_offset + 2])[0]
            mod_start = struct.unpack("<H", pbag_data[bag_offset + 2:bag_offset + 4])[0]

            # 次のバッグのインデックスを取得
            next_bag_offset = (bag_idx + 1) * 4
            gen_end = struct.unpack("<H", pbag_data[next_bag_offset:next_bag_offset + 2])[0]
            mod_end = struct.unpack("<H", pbag_data[next_bag_offset + 2:next_bag_offset + 4])[0]

            zone_mods = mod_end - mod_start
            zone_gens = gen_end - gen_start

            if zone_mods == 0 or zone_gens == 0:
                preset_issues.append({
                    "preset": f"{bank:03d}:{preset_num:03d} {name}",
                    "zone": z,
                    "mods": zone_mods,
                    "gens": zone_gens
                })

    if preset_issues:
        print("Presets with empty modulators or generators:")
        for issue in preset_issues:
            print(f"  {issue["preset"]} zone {issue["zone"]}: {issue["mods"]} mods, {issue["gens"]} gens")
        print()

    return num_mods, num_gens


def analyze_instrument_zones(pdta):
    """インストゥルメントゾーンのモジュレータ・ジェネレータを詳細分析"""
    inst_data = pdta["inst"]
    ibag_data = pdta["ibag"]
    imod_data = pdta["imod"]
    igen_data = pdta["igen"]

    num_insts = len(inst_data) // 22 - 1  # 最後はターミネータ
    num_bags = len(ibag_data) // 4 - 1
    num_mods = len(imod_data) // 10
    num_gens = len(igen_data) // 4

    print(f"Instrument Analysis:")
    print(f"  Instruments: {num_insts}")
    print(f"  Bags: {num_bags}")
    print(f"  Modulators: {num_mods}")
    print(f"  Generators: {num_gens}")
    print()

    # 各インストゥルメントのゾーンを確認
    inst_issues = []

    for i in range(num_insts):
        offset = i * 22
        name = inst_data[offset:offset + 20].rstrip(b"\x00").decode("ascii", errors="ignore")
        bag_start = struct.unpack("<H", inst_data[offset + 20:offset + 22])[0]

        # 次のインストゥルメントのbag_startを取得
        next_offset = (i + 1) * 22
        bag_end = struct.unpack("<H", inst_data[next_offset + 20:next_offset + 22])[0]

        num_zones = bag_end - bag_start

        # 各ゾーンのモジュレータとジェネレータ数を確認
        for z in range(num_zones):
            bag_idx = bag_start + z
            bag_offset = bag_idx * 4

            gen_start = struct.unpack("<H", ibag_data[bag_offset:bag_offset + 2])[0]
            mod_start = struct.unpack("<H", ibag_data[bag_offset + 2:bag_offset + 4])[0]

            # 次のバッグのインデックスを取得
            next_bag_offset = (bag_idx + 1) * 4
            gen_end = struct.unpack("<H", ibag_data[next_bag_offset:next_bag_offset + 2])[0]
            mod_end = struct.unpack("<H", ibag_data[next_bag_offset + 2:next_bag_offset + 4])[0]

            zone_mods = mod_end - mod_start
            zone_gens = gen_end - gen_start

            if zone_mods == 0 or zone_gens == 0:
                inst_issues.append({
                    "instrument": name,
                    "zone": z,
                    "mods": zone_mods,
                    "gens": zone_gens
                })

    if inst_issues:
        print("Instruments with empty modulators or generators:")
        for issue in inst_issues:
            print(f"  {issue["instrument"]} zone {issue["zone"]}: {issue["mods"]} mods, {issue["gens"]} gens")
        print()

    return num_mods, num_gens


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <sf2_file>")
        sys.exit(1)

    sf2_path = sys.argv[1]
    print(f"Analyzing: {sf2_path}")
    print("=" * 70)
    print()

    pdta = parse_sf2_pdta(sf2_path)

    pmod_count, pgen_count = analyze_preset_zones(pdta)
    imod_count, igen_count = analyze_instrument_zones(pdta)

    print("=" * 70)
    print("Summary:")
    print(f"  Preset Modulators: {pmod_count}")
    print(f"  Preset Generators: {pgen_count}")
    print(f"  Instrument Modulators: {imod_count}")
    print(f"  Instrument Generators: {igen_count}")


if __name__ == "__main__":
    main()
