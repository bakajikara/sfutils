"""
FLACファイルから実際のオーディオサンプルをデコードして比較
"""
import subprocess
import struct
import sys
import os


def decode_flac_to_raw(flac_file, output_file):
    """FLACファイルをRAW PCMにデコード"""
    # flac コマンドを使用してデコード
    cmd = [
        "flac",
        "-d",
        "-f",
        "--force-raw-format",
        "--endian=little",
        "--sign=signed",
        flac_file,
        "-o", output_file
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"flacコマンドが見つかりません。代替方法を試します...")
            return False
        return True
    except FileNotFoundError:
        print(f"flacコマンドが見つかりません。")
        return False


def analyze_raw_pcm(file1, file2):
    """RAW PCMファイルを比較"""
    with open(file1, "rb") as f:
        data1 = f.read()
    with open(file2, "rb") as f:
        data2 = f.read()

    print(f"\n{"=" * 60}")
    print("PCMデータ比較")
    print(f"{"=" * 60}")
    print(f"ファイル1サイズ: {len(data1)} バイト ({len(data1) // 2} サンプル)")
    print(f"ファイル2サイズ: {len(data2)} バイト ({len(data2) // 2} サンプル)")

    if len(data1) != len(data2):
        print(f"⚠ サイズが異なります！")
        return

    # 16bitサンプルとして解析
    samples1 = []
    samples2 = []

    for i in range(0, len(data1), 2):
        s1 = struct.unpack("<h", data1[i:i + 2])[0]
        s2 = struct.unpack("<h", data2[i:i + 2])[0]
        samples1.append(s1)
        samples2.append(s2)

    # 違いを検出
    differences = []
    for i in range(len(samples1)):
        if samples1[i] != samples2[i]:
            differences.append((i, samples1[i], samples2[i]))

    if differences:
        print(f"\n⚠ {len(differences)} サンプルの違いが見つかりました ({len(differences) / len(samples1) * 100:.2f}%)")
        print(f"\n最初の20サンプルの違い:")
        for i, (idx, s1, s2) in enumerate(differences[:20]):
            diff = s2 - s1
            time_sec = idx / 44100
            print(f"  サンプル {idx:8d} ({time_sec:7.4f}秒): {s1:6d} -> {s2:6d} (差分: {diff:+6d})")

        # 統計情報
        diffs = [s2 - s1 for _, s1, s2 in differences]
        max_diff = max(abs(d) for d in diffs)
        avg_diff = sum(abs(d) for d in diffs) / len(diffs)

        print(f"\n差分の統計:")
        print(f"  最大差分: {max_diff}")
        print(f"  平均差分: {avg_diff:.2f}")
        print(f"  最小差分値: {min(diffs)}")
        print(f"  最大差分値: {max(diffs)}")

        # 差分の分布
        small_diffs = [d for d in diffs if abs(d) <= 1]
        medium_diffs = [d for d in diffs if 1 < abs(d) <= 10]
        large_diffs = [d for d in diffs if abs(d) > 10]

        print(f"\n差分の分布:")
        print(f"  微小差分 (±1以内): {len(small_diffs)} ({len(small_diffs) / len(diffs) * 100:.1f}%)")
        print(f"  中程度差分 (±2-10): {len(medium_diffs)} ({len(medium_diffs) / len(diffs) * 100:.1f}%)")
        print(f"  大きな差分 (±10超): {len(large_diffs)} ({len(large_diffs) / len(diffs) * 100:.1f}%)")

    else:
        print(f"\n✓ すべてのサンプルが完全に一致しています")


# 試しにsoundfileライブラリを使う方法
try:
    import soundfile as sf

    def compare_with_soundfile(file1, file2):
        """soundfileを使用してFLACファイルを直接比較"""
        print("\nsoundfileライブラリを使用してデコード中...")

        data1, sr1 = sf.read(file1, dtype="int16")
        data2, sr2 = sf.read(file2, dtype="int16")

        print(f"\n{"=" * 60}")
        print("soundfileによる比較")
        print(f"{"=" * 60}")
        print(f"ファイル1: {len(data1)} サンプル, {sr1} Hz")
        print(f"ファイル2: {len(data2)} サンプル, {sr2} Hz")

        if len(data1) != len(data2):
            print(f"⚠ サンプル数が異なります！")
            return

        # 違いを検出
        differences = []
        for i in range(len(data1)):
            if data1[i] != data2[i]:
                differences.append((i, data1[i], data2[i]))

        if differences:
            print(f"\n⚠ {len(differences)} サンプルの違いが見つかりました ({len(differences) / len(data1) * 100:.2f}%)")
            print(f"\n最初の20サンプルの違い:")
            for i, (idx, s1, s2) in enumerate(differences[:20]):
                diff = s2 - s1
                time_sec = idx / sr1
                print(f"  サンプル {idx:8d} ({time_sec:7.4f}秒): {s1:6d} -> {s2:6d} (差分: {diff:+6d})")

            # 統計情報
            diffs = [int(s2) - int(s1) for _, s1, s2 in differences]
            max_diff = max(abs(d) for d in diffs)
            avg_diff = sum(abs(d) for d in diffs) / len(diffs)

            print(f"\n差分の統計:")
            print(f"  最大差分: {max_diff}")
            print(f"  平均差分: {avg_diff:.2f}")
            print(f"  最小差分値: {min(diffs)}")
            print(f"  最大差分値: {max(diffs)}")

            # 差分の分布
            small_diffs = [d for d in diffs if abs(d) <= 1]
            medium_diffs = [d for d in diffs if 1 < abs(d) <= 10]
            large_diffs = [d for d in diffs if abs(d) > 10]

            print(f"\n差分の分布:")
            print(f"  微小差分 (±1以内): {len(small_diffs)} ({len(small_diffs) / len(diffs) * 100:.1f}%)")
            print(f"  中程度差分 (±2-10): {len(medium_diffs)} ({len(medium_diffs) / len(diffs) * 100:.1f}%)")
            print(f"  大きな差分 (±10超): {len(large_diffs)} ({len(large_diffs) / len(diffs) * 100:.1f}%)")

            # 差分があるサンプルの位置分布
            print(f"\n差分があるサンプルの位置:")
            first_diff = differences[0][0]
            last_diff = differences[-1][0]
            print(f"  最初の差分: サンプル {first_diff} ({first_diff / sr1:.4f}秒)")
            print(f"  最後の差分: サンプル {last_diff} ({last_diff / sr1:.4f}秒)")

            # 10区間に分けて分布を確認
            section_size = len(data1) // 10
            for section in range(10):
                start = section * section_size
                end = start + section_size
                section_diffs = [d for d in differences if start <= d[0] < end]
                print(f"  区間 {section} ({start / sr1:.2f}-{end / sr1:.2f}秒): {len(section_diffs)} 個の差分")
        else:
            print(f"\n✓ すべてのサンプルが完全に一致しています")

    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    print("soundfileライブラリがインストールされていません")

if __name__ == "__main__":
    file1 = r"C:\Users\tokib2a3\Desktop\sf\MSBasic_test\samples\58_74_HALFCRASH1.flac"
    file2 = r"C:\Users\tokib2a3\Desktop\sf\MSBasic_test2\samples\58_74_HALFCRASH1.flac"

    if HAS_SOUNDFILE:
        compare_with_soundfile(file1, file2)
    else:
        print("\nsoundfileをインストールしています...")
        print("pip install soundfile を実行してください")
