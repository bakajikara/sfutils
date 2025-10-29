"""
2つのFLACファイルの違いを分析するツール
"""
import sys
import struct
import hashlib


def read_flac_file(filepath):
    """FLACファイルを読み込んで分析"""
    with open(filepath, "rb") as f:
        data = f.read()

    print(f"\n{"=" * 60}")
    print(f"ファイル: {filepath}")
    print(f"{"=" * 60}")
    print(f"ファイルサイズ: {len(data)} バイト")
    print(f"MD5ハッシュ: {hashlib.md5(data).hexdigest()}")
    print(f"SHA256ハッシュ: {hashlib.sha256(data).hexdigest()}")

    # FLACヘッダーチェック
    if data[:4] != b"fLaC":
        print("警告: 有効なFLACファイルではありません")
        return None

    print("FLACヘッダー: OK")

    # メタデータブロックを解析
    pos = 4
    block_num = 0

    while pos < len(data):
        if pos + 4 > len(data):
            break

        # メタデータブロックヘッダー
        header_byte = data[pos]
        is_last = (header_byte & 0x80) != 0
        block_type = header_byte & 0x7F

        block_length = struct.unpack(">I", b"\x00" + data[pos + 1:pos + 4])[0]

        block_type_names = {
            0: "STREAMINFO",
            1: "PADDING",
            2: "APPLICATION",
            3: "SEEKTABLE",
            4: "VORBIS_COMMENT",
            5: "CUESHEET",
            6: "PICTURE"
        }

        type_name = block_type_names.get(block_type, f"UNKNOWN({block_type})")

        print(f"\nメタデータブロック #{block_num}:")
        print(f"  タイプ: {type_name}")
        print(f"  サイズ: {block_length} バイト")
        print(f"  最後のブロック: {"はい" if is_last else "いいえ"}")

        # STREAMINFOの詳細を表示
        if block_type == 0 and block_length >= 34:
            streaminfo = data[pos + 4:pos + 4 + block_length]
            min_blocksize = struct.unpack(">H", streaminfo[0:2])[0]
            max_blocksize = struct.unpack(">H", streaminfo[2:4])[0]
            min_framesize = struct.unpack(">I", b"\x00" + streaminfo[4:7])[0]
            max_framesize = struct.unpack(">I", b"\x00" + streaminfo[7:10])[0]

            # サンプルレート、チャンネル数、ビット深度を解析
            val = struct.unpack(">Q", streaminfo[10:18])[0]
            sample_rate = (val >> 44) & 0xFFFFF
            channels = ((val >> 41) & 0x7) + 1
            bits_per_sample = ((val >> 36) & 0x1F) + 1
            total_samples = val & 0xFFFFFFFFF

            md5_signature = streaminfo[18:34].hex()

            print(f"  最小ブロックサイズ: {min_blocksize}")
            print(f"  最大ブロックサイズ: {max_blocksize}")
            print(f"  最小フレームサイズ: {min_framesize}")
            print(f"  最大フレームサイズ: {max_framesize}")
            print(f"  サンプルレート: {sample_rate} Hz")
            print(f"  チャンネル数: {channels}")
            print(f"  ビット深度: {bits_per_sample} bits")
            print(f"  総サンプル数: {total_samples}")
            print(f"  再生時間: {total_samples / sample_rate:.3f} 秒")
            print(f"  オーディオMD5: {md5_signature}")

        # VORBIS_COMMENTの詳細を表示
        elif block_type == 4:
            vorbis_data = data[pos + 4:pos + 4 + block_length]
            if len(vorbis_data) >= 4:
                vendor_length = struct.unpack("<I", vorbis_data[0:4])[0]
                if len(vorbis_data) >= 4 + vendor_length + 4:
                    vendor_string = vorbis_data[4:4 + vendor_length].decode("utf-8", errors="replace")
                    print(f"  ベンダー: {vendor_string}")

                    comment_pos = 4 + vendor_length
                    num_comments = struct.unpack("<I", vorbis_data[comment_pos:comment_pos + 4])[0]
                    print(f"  コメント数: {num_comments}")

                    comment_pos += 4
                    for i in range(num_comments):
                        if comment_pos + 4 > len(vorbis_data):
                            break
                        comment_length = struct.unpack("<I", vorbis_data[comment_pos:comment_pos + 4])[0]
                        comment_pos += 4
                        if comment_pos + comment_length > len(vorbis_data):
                            break
                        comment = vorbis_data[comment_pos:comment_pos + comment_length].decode("utf-8", errors="replace")
                        print(f"    {comment}")
                        comment_pos += comment_length

        pos += 4 + block_length
        block_num += 1

        if is_last:
            print(f"\nオーディオフレーム開始位置: {pos} バイト")
            print(f"オーディオデータサイズ: {len(data) - pos} バイト")
            break

    return data


def compare_files(file1, file2):
    """2つのファイルをバイト単位で比較"""
    data1 = read_flac_file(file1)
    data2 = read_flac_file(file2)

    if data1 is None or data2 is None:
        return

    print(f"\n{"=" * 60}")
    print("比較結果")
    print(f"{"=" * 60}")

    if len(data1) != len(data2):
        print(f"⚠ ファイルサイズが異なります:")
        print(f"  ファイル1: {len(data1)} バイト")
        print(f"  ファイル2: {len(data2)} バイト")
        print(f"  差分: {abs(len(data1) - len(data2))} バイト")
    else:
        print(f"✓ ファイルサイズは同じです: {len(data1)} バイト")

    # バイト単位での違いを検出
    min_len = min(len(data1), len(data2))
    differences = []

    for i in range(min_len):
        if data1[i] != data2[i]:
            differences.append(i)

    if differences:
        print(f"\n⚠ {len(differences)} バイトの違いが見つかりました")
        print(f"  最初の違い: オフセット {differences[0]} (0x{differences[0]:X})")
        print(f"  最後の違い: オフセット {differences[-1]} (0x{differences[-1]:X})")

        # 最初の10箇所の違いを詳細表示
        print(f"\n最初の10箇所の違い:")
        for i, pos in enumerate(differences[:10]):
            print(f"  オフセット {pos:8d} (0x{pos:08X}): 0x{data1[pos]:02X} != 0x{data2[pos]:02X}")

        # 違いの分布を確認
        print(f"\n違いの分布:")
        if differences[0] < 4:
            print("  FLACマジックナンバー内に違いがあります")

        # メタデータ領域内の違い
        metadata_diffs = [d for d in differences if d < 1000]  # 概算
        if metadata_diffs:
            print(f"  メタデータ領域内の違い: {len(metadata_diffs)} バイト")

        # オーディオデータ領域内の違い
        audio_diffs = [d for d in differences if d >= 1000]  # 概算
        if audio_diffs:
            print(f"  オーディオデータ領域内の違い: {len(audio_diffs)} バイト")
    else:
        print(f"\n✓ ファイルは完全に同一です")


if __name__ == "__main__":
    file1 = r"C:\Users\tokib2a3\Desktop\sf\MSBasic_test\samples\58_74_HALFCRASH1.flac"
    file2 = r"C:\Users\tokib2a3\Desktop\sf\MSBasic_test2\samples\58_74_HALFCRASH1.flac"

    compare_files(file1, file2)
