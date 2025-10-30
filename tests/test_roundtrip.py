#!/usr/bin/env python3
"""
ラウンドトリップテスト - SF2ファイルの完全性を検証

MuseScore_General_HQ.sf2をデコンパイル→コンパイルして、
バイナリレベルで完全一致することを確認します。

完全一致しない場合は、実用上の等価性を検証します。
等価性チェックでは以下を無視します:
- サンプル、インストゥルメント、プリセットの順序
- 内部的なサンプル名などの名称の微差
- 順序の違いによるID/offset値の差異
"""

import sys
import hashlib
import shutil
from pathlib import Path
from sfutils.decompiler import SF2Decompiler
from sfutils.compiler import SF2Compiler
from test_equivalence import SF2EquivalenceChecker


def calculate_md5(filepath):
    """ファイルのMD5ハッシュを計算"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def calculate_sha256(filepath):
    """ファイルのSHA256ハッシュを計算"""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def main():
    print("=" * 80)
    print("SF2 Roundtrip Test - MuseScore_General_HQ.sf2")
    print("=" * 80)

    # ファイルパス設定
    original_sf2 = Path("MuseScore_General_HQ.sf2")
    temp_dir = Path("temp_roundtrip_test")
    rebuilt_sf2 = Path("temp_rebuilt.sf2")

    # オリジナルファイルの存在確認
    if not original_sf2.exists():
        print(f"❌ Error: {original_sf2} が見つかりません")
        sys.exit(1)

    print(f"\n📁 Original file: {original_sf2}")
    print(f"   Size: {original_sf2.stat().st_size:,} bytes")

    try:
        # Step 1: デコンパイル
        print(f"\n🔓 Step 1: デコンパイル中 → {temp_dir}")
        if temp_dir.exists():
            print(f"   既存のディレクトリを削除中...")
            shutil.rmtree(temp_dir)

        decompiler = SF2Decompiler(str(original_sf2), str(temp_dir))
        decompiler.decompile()
        print(f"   ✓ デコンパイル完了")

        # デコンパイルされたファイル数を表示
        samples = list((temp_dir / "samples").glob("*.wav"))
        instruments = list((temp_dir / "instruments").glob("*.json"))
        presets = list((temp_dir / "presets").glob("*.json"))
        print(f"   - Samples: {len(samples)}")
        print(f"   - Instruments: {len(instruments)}")
        print(f"   - Presets: {len(presets)}")

        # Step 2: コンパイル
        print(f"\n🔒 Step 2: コンパイル中 → {rebuilt_sf2}")
        if rebuilt_sf2.exists():
            print(f"   既存のファイルを削除中...")
            rebuilt_sf2.unlink()

        compiler = SF2Compiler(str(temp_dir), str(rebuilt_sf2))
        compiler.compile()
        print(f"   ✓ コンパイル完了")
        print(f"   Size: {rebuilt_sf2.stat().st_size:,} bytes")

        # Step 3: バイナリ完全一致チェック
        print(f"\n🔬 Step 3: バイナリ完全一致チェック中...")

        original_md5 = calculate_md5(original_sf2)
        rebuilt_md5 = calculate_md5(rebuilt_sf2)

        print(f"   MD5ハッシュ:")
        print(f"   - Original: {original_md5}")
        print(f"   - Rebuilt:  {rebuilt_md5}")

        is_identical = (original_md5 == rebuilt_md5)

        if is_identical:
            print(f"   ✅ 完全一致！")
        else:
            print(f"   ❌ バイナリレベルでは不一致")

        # Step 4: 等価性チェック（完全一致しなかった場合のみ）
        is_equivalent = False
        if not is_identical:
            print(f"\n🔍 Step 4: 等価性チェック中...")
            print(f"   （バイナリ完全一致しなかったため、実用上の等価性を検証します）")
            print()

            checker = SF2EquivalenceChecker(str(original_sf2), str(rebuilt_sf2))
            is_equivalent = checker.check()
        else:
            print(f"\n✨ バイナリ完全一致のため、等価性チェックはスキップします")
            is_equivalent = True

        # 最終結果
        print("\n" + "=" * 80)
        if is_identical:
            print("🎉 テスト成功！完全可逆変換が確認されました！")
            print("   （バイナリレベルで完全一致）")
            print("=" * 80)
            success = True
        elif is_equivalent:
            print("✅ テスト成功！等価なSF2ファイルが生成されました")
            print("   （バイナリは異なるが、実用上は等価）")
            print("=" * 80)
            success = True
        else:
            print("❌ テスト失敗：等価性の問題が検出されました")
            print("=" * 80)
            success = False

        # クリーンアップするか確認
        print(f"\n🧹 クリーンアップ:")
        print(f"   一時ファイル: {temp_dir}, {rebuilt_sf2}")

        if success:
            response = input("   一時ファイルを削除しますか? [Y/n]: ").strip().lower()
            if response in ['', 'y', 'yes']:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    print(f"   ✓ {temp_dir} を削除しました")
                if rebuilt_sf2.exists():
                    rebuilt_sf2.unlink()
                    print(f"   ✓ {rebuilt_sf2} を削除しました")
            else:
                print(f"   一時ファイルを保持します")
        else:
            print(f"   テスト失敗のため、一時ファイルを保持します（デバッグ用）")

        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
