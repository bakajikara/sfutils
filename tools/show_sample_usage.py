#!/usr/bin/env python3
"""
サウンドフォントのサンプル使用状況を表示するツール

プリセット番号順に、各プリセットが参照するサンプルを表示します。
既に表示されたサンプルは重複して表示せず、最初に参照したプリセットにのみ表示されます。
最後に、どのプリセットからも参照されていないサンプルを表示します。
"""

from sfutils.parser import SoundFontParser
import sys
import os
import argparse

# sfutilsモジュールをインポートできるようにパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def get_samples_for_instrument(parser, inst_idx, seen_samples):
    """
    インストゥルメントが使用するサンプルのリストを取得します。

    Args:
        parser: SoundFontParserインスタンス
        inst_idx: インストゥルメントのインデックス
        seen_samples: 既に表示済みのサンプルインデックスのセット

    Returns:
        新たに見つかったサンプルのリスト [(sample_idx, sample_name), ...]
    """
    zones = parser.get_instrument_zones(inst_idx)
    samples = []

    for zone in zones:
        # ジェネレータからsampleID (oper=53)を探す
        for gen in zone["generators"]:
            if gen["oper"] == 53:  # sampleID
                sample_idx = gen["amount"]

                # まだ表示していないサンプルのみ追加
                if sample_idx not in seen_samples:
                    sample_headers = parser.get_sample_headers()
                    if 0 <= sample_idx < len(sample_headers):
                        sample_name = sample_headers[sample_idx]["name"]
                        samples.append((sample_idx, sample_name))
                        seen_samples.add(sample_idx)

    return samples


def get_samples_for_preset(parser, preset_idx, seen_samples):
    """
    プリセットが使用するサンプルのリストを取得します。

    Args:
        parser: SoundFontParserインスタンス
        preset_idx: プリセットのインデックス
        seen_samples: 既に表示済みのサンプルインデックスのセット

    Returns:
        新たに見つかったサンプルのリスト [(sample_idx, sample_name), ...]
    """
    zones = parser.get_preset_zones(preset_idx)
    all_samples = []

    for zone in zones:
        # ジェネレータからinstrument (oper=41)を探す
        for gen in zone["generators"]:
            if gen["oper"] == 41:  # instrument
                inst_idx = gen["amount"]
                samples = get_samples_for_instrument(parser, inst_idx, seen_samples)
                all_samples.extend(samples)

    return all_samples


def show_sample_usage(sf2_path, mode="preset"):
    """
    サウンドフォントのサンプル使用状況を表示します。

    Args:
        sf2_path: サウンドフォントファイルのパス
        mode: 表示モード ("preset" または "instrument")
    """
    parser = SoundFontParser(sf2_path)
    parser.parse()

    preset_headers = parser.get_preset_headers()
    instrument_headers = parser.get_instrument_headers()
    sample_headers = parser.get_sample_headers()

    seen_samples = set()

    print(f"サウンドフォント: {sf2_path}")
    print(f"総プリセット数: {len(preset_headers)}")
    print(f"総インストゥルメント数: {len(instrument_headers)}")
    print(f"総サンプル数: {len(sample_headers)}")
    print("=" * 80)
    print()

    if mode == "instrument":
        # インストゥルメント順にサンプルを表示
        for inst_idx, inst in enumerate(instrument_headers):
            samples = get_samples_for_instrument(parser, inst_idx, seen_samples)

            if samples:
                inst_name = inst["name"]
                print(f"インストゥルメント [{inst_idx:04d}] - {inst_name}")
                for sample_idx, sample_name in samples:
                    print(f"  [{sample_idx:04d}] {sample_name}")
                print()
    else:
        # プリセット順にサンプルを表示
        # プリセットを(bank, preset)の順でソート
        sorted_presets = sorted(
            enumerate(preset_headers),
            key=lambda x: (x[1]["bank"], x[1]["preset"])
        )

        for preset_idx, preset in sorted_presets:
            samples = get_samples_for_preset(parser, preset_idx, seen_samples)

            if samples:
                bank = preset["bank"]
                preset_num = preset["preset"]
                preset_name = preset["name"]

                print(f"プリセット {bank:03d}:{preset_num:03d} - {preset_name}")
                for sample_idx, sample_name in samples:
                    print(f"  [{sample_idx:04d}] {sample_name}")
                print()

    # 未使用サンプルを表示
    all_sample_indices = set(range(len(sample_headers)))
    unused_samples = all_sample_indices - seen_samples

    if unused_samples:
        print("=" * 80)
        print("未使用サンプル:")
        print("=" * 80)
        for sample_idx in sorted(unused_samples):
            sample_name = sample_headers[sample_idx]["name"]
            sample_type = sample_headers[sample_idx]["sample_type"]
            print(f"  [{sample_idx:04d}] {sample_name} (type={sample_type})")
        print()
        print(f"未使用サンプル数: {len(unused_samples)}")
    else:
        print("=" * 80)
        print("すべてのサンプルが使用されています。")
        print("=" * 80)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="サウンドフォントのサンプル使用状況を表示します。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # プリセットごとに表示 (デフォルト)
  python show_sample_usage.py soundfont.sf2

  # インストゥルメントごとに表示
  python show_sample_usage.py soundfont.sf2 --mode instrument
  python show_sample_usage.py soundfont.sf2 -m i
        """
    )

    parser.add_argument(
        "sf2_file",
        help="SoundFontファイルのパス"
    )

    parser.add_argument(
        "-m", "--mode",
        choices=["preset", "instrument", "p", "i"],
        default="preset",
        help="表示モード: preset (p) = プリセットごと, instrument (i) = インストゥルメントごと (デフォルト: preset)"
    )

    args = parser.parse_args()

    # モードの短縮形を正式名に変換
    mode = args.mode
    if mode == "p":
        mode = "preset"
    elif mode == "i":
        mode = "instrument"

    sf2_path = args.sf2_file

    if not os.path.exists(sf2_path):
        print(f"エラー: ファイルが見つかりません: {sf2_path}")
        sys.exit(1)

    try:
        show_sample_usage(sf2_path, mode)
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
