#!/usr/bin/env python3
"""
バラしたサウンドフォント内の未参照ファイルを検出するスクリプト

サウンドフォントから展開されたファイル構造の中で、
instrumentsやpresetsから参照されていないsampleファイルを見つけます。
"""

import os
import json
import sys
from pathlib import Path


def load_json_file(filepath):
    """JSONファイルを読み込む"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"警告: {filepath} の読み込みに失敗: {e}")
        return None


def extract_sample_references(data, referenced_samples):
    """
    JSON データから参照されているサンプルIDを抽出
    instrumentsとpresetsで使われているsampleIDを収集
    """
    if isinstance(data, dict):
        # sampleID フィールドを探す
        if 'sampleID' in data:
            sample_id = data['sampleID']
            referenced_samples.add(sample_id)

        # 再帰的に辞書内を探索
        for value in data.values():
            extract_sample_references(value, referenced_samples)

    elif isinstance(data, list):
        # リスト内の各要素を探索
        for item in data:
            extract_sample_references(item, referenced_samples)


def check_unreferenced_files(root_dir):
    """
    指定されたディレクトリ内の未参照ファイルをチェック

    Args:
        root_dir: サウンドフォントが展開されたルートディレクトリ
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"エラー: ディレクトリ {root_dir} が存在しません")
        return

    # 各サブディレクトリのパス
    instruments_dir = root_path / "instruments"
    presets_dir = root_path / "presets"
    samples_dir = root_path / "samples"

    # ディレクトリの存在確認
    if not samples_dir.exists():
        print(f"エラー: samples ディレクトリが存在しません: {samples_dir}")
        return

    print(f"サウンドフォントディレクトリ: {root_dir}")
    print("=" * 70)

    # 参照されているサンプルIDを格納するセット
    referenced_samples = set()

    # 1. instrumentsディレクトリからサンプル参照を収集
    if instruments_dir.exists():
        instrument_files = list(instruments_dir.glob("*.json"))
        print(f"\nInstrumentsをスキャン中... ({len(instrument_files)} ファイル)")

        for inst_file in instrument_files:
            data = load_json_file(inst_file)
            if data:
                extract_sample_references(data, referenced_samples)
    else:
        print(f"警告: instruments ディレクトリが存在しません: {instruments_dir}")

    # 2. presetsディレクトリからサンプル参照を収集
    if presets_dir.exists():
        preset_files = list(presets_dir.glob("*.json"))
        print(f"Presetsをスキャン中... ({len(preset_files)} ファイル)")

        for preset_file in preset_files:
            data = load_json_file(preset_file)
            if data:
                extract_sample_references(data, referenced_samples)
    else:
        print(f"警告: presets ディレクトリが存在しません: {presets_dir}")

    print(f"\n参照されているサンプル数: {len(referenced_samples)}")

    # 3. samplesディレクトリ内の実際のサンプルファイルを取得
    # .flac と .json のペアを考慮
    sample_files = {}  # サンプル名 -> [.flac, .json] のマッピング

    for sample_file in samples_dir.iterdir():
        if sample_file.is_file():
            name = sample_file.stem  # 拡張子を除いたファイル名
            ext = sample_file.suffix

            if name not in sample_files:
                sample_files[name] = []
            sample_files[name].append(ext)

    print(f"検出されたサンプル数: {len(sample_files)}")

    # 4. 未参照のサンプルを検出
    unreferenced_samples = []

    for sample_name, extensions in sample_files.items():
        if sample_name not in referenced_samples:
            unreferenced_samples.append((sample_name, extensions))

    # 5. 結果を表示
    print("\n" + "=" * 70)
    if unreferenced_samples:
        print(f"\n❌ 未参照のサンプルファイル: {len(unreferenced_samples)} 個")
        print("-" * 70)

        # ソートして表示
        unreferenced_samples.sort()
        for sample_name, extensions in unreferenced_samples:
            ext_str = ", ".join(extensions)
            print(f"  {sample_name} ({ext_str})")

        # サマリー
        print("\n" + "=" * 70)
        print(f"合計: {len(unreferenced_samples)} 個の未参照サンプル")

        # サイズを計算（オプション）
        total_size = 0
        for sample_name, _ in unreferenced_samples:
            for ext in ['.flac', '.json']:
                filepath = samples_dir / f"{sample_name}{ext}"
                if filepath.exists():
                    total_size += filepath.stat().st_size

        if total_size > 0:
            size_mb = total_size / (1024 * 1024)
            print(f"未参照ファイルの合計サイズ: {size_mb:.2f} MB")
    else:
        print("\n✅ すべてのサンプルファイルが参照されています！")

    print("=" * 70)

    # 6. 逆に、参照されているが存在しないサンプルもチェック
    missing_samples = []
    for ref_sample in referenced_samples:
        if ref_sample not in sample_files:
            missing_samples.append(ref_sample)

    if missing_samples:
        print(f"\n⚠️  参照されているが存在しないサンプル: {len(missing_samples)} 個")
        print("-" * 70)
        missing_samples.sort()
        for sample_name in missing_samples:
            print(f"  {sample_name}")
        print("=" * 70)


def main():
    """メイン処理"""
    if len(sys.argv) < 2:
        print("使用方法: python check_unreferenced.py <サウンドフォントディレクトリ>")
        print("\n例:")
        print("  python check_unreferenced.py MSBasic")
        print("  python check_unreferenced.py MSBasic_test")
        sys.exit(1)

    target_dir = sys.argv[1]
    check_unreferenced_files(target_dir)


if __name__ == "__main__":
    main()
