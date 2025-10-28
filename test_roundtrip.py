#!/usr/bin/env python3
"""
ラウンドトリップテスト - SF2ファイルの完全可逆性を検証

MuseScore_General_HQ.sf2をデコンパイル→コンパイルして、
バイナリレベルで完全一致することを確認します。
"""

import sys
import hashlib
import shutil
from pathlib import Path
from sf2_decompiler import SF2Decompiler
from sf2_compiler import SF2Compiler


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


def compare_files_binary(file1, file2):
    """2つのファイルをバイナリレベルで比較"""
    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        data1 = f1.read()
        data2 = f2.read()

        if data1 == data2:
            return True, None

        # 差異の詳細を調べる
        min_len = min(len(data1), len(data2))
        first_diff = None

        for i in range(min_len):
            if data1[i] != data2[i]:
                first_diff = i
                break

        diff_info = {
            "size1": len(data1),
            "size2": len(data2),
            "first_diff": first_diff,
            "size_match": len(data1) == len(data2)
        }

        return False, diff_info


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
        # Step 1: オリジナルファイルのハッシュ計算
        print("\n🔍 Step 1: オリジナルファイルのハッシュを計算中...")
        original_md5 = calculate_md5(original_sf2)
        original_sha256 = calculate_sha256(original_sf2)
        print(f"   MD5:    {original_md5}")
        print(f"   SHA256: {original_sha256}")

        # Step 2: デコンパイル
        print(f"\n🔓 Step 2: デコンパイル中 → {temp_dir}")
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

        # Step 3: コンパイル
        print(f"\n🔒 Step 3: コンパイル中 → {rebuilt_sf2}")
        if rebuilt_sf2.exists():
            print(f"   既存のファイルを削除中...")
            rebuilt_sf2.unlink()

        compiler = SF2Compiler(str(temp_dir), str(rebuilt_sf2))
        compiler.compile()
        print(f"   ✓ コンパイル完了")
        print(f"   Size: {rebuilt_sf2.stat().st_size:,} bytes")

        # Step 4: リビルドファイルのハッシュ計算
        print("\n🔍 Step 4: リビルドファイルのハッシュを計算中...")
        rebuilt_md5 = calculate_md5(rebuilt_sf2)
        rebuilt_sha256 = calculate_sha256(rebuilt_sf2)
        print(f"   MD5:    {rebuilt_md5}")
        print(f"   SHA256: {rebuilt_sha256}")

        # Step 5: 比較
        print("\n🔬 Step 5: バイナリ比較中...")

        # ファイルサイズ比較
        orig_size = original_sf2.stat().st_size
        rebu_size = rebuilt_sf2.stat().st_size

        print(f"\n   ファイルサイズ:")
        print(f"   - Original: {orig_size:,} bytes")
        print(f"   - Rebuilt:  {rebu_size:,} bytes")
        print(f"   - Diff:     {rebu_size - orig_size:+,} bytes")

        if orig_size == rebu_size:
            print(f"   ✅ サイズ一致")
        else:
            print(f"   ❌ サイズ不一致")

        # ハッシュ比較
        print(f"\n   MD5ハッシュ:")
        if original_md5 == rebuilt_md5:
            print(f"   ✅ 一致: {original_md5}")
        else:
            print(f"   ❌ 不一致")
            print(f"   - Original: {original_md5}")
            print(f"   - Rebuilt:  {rebuilt_md5}")

        print(f"\n   SHA256ハッシュ:")
        if original_sha256 == rebuilt_sha256:
            print(f"   ✅ 一致: {original_sha256}")
        else:
            print(f"   ❌ 不一致")
            print(f"   - Original: {original_sha256}")
            print(f"   - Rebuilt:  {rebuilt_sha256}")

        # バイト単位での比較
        # print(f"\n   バイト単位での比較:")
        # is_identical, diff_info = compare_files_binary(original_sf2, rebuilt_sf2)

        # if is_identical:
        #     print(f"   ✅ 完全一致 - バイナリレベルで同一です！")
        # else:
        #     print(f"   ❌ 差異あり")
        #     if diff_info["first_diff"] is not None:
        #         print(f"   - 最初の差異: バイト {diff_info['first_diff']:,}")
        #     if not diff_info["size_match"]:
        #         print(f"   - サイズ差: {diff_info['size2'] - diff_info['size1']:+,} bytes")

        # 最終結果
        print("\n" + "=" * 80)
        if original_md5 == rebuilt_md5 and original_sha256 == rebuilt_sha256:
            print("🎉 テスト成功！完全可逆変換が確認されました！")
            print("=" * 80)
            success = True
        else:
            print("❌ テスト失敗：差異が検出されました")
            print("=" * 80)
            print("\n詳細な差分解析を実行するには:")
            print(f"  python binary_compare.py {original_sf2} {rebuilt_sf2}")
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
