import sys
import os


def hexdump(data, start_offset=0):
    """
    指定されたバイナリデータを16進ダンプ形式で表示するヘルパー関数。
    """
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]

        # 1. オフセット
        offset_str = f"{start_offset + i:08X}"

        # 2. 16進データ
        hex_part = " ".join(f"{b:02X}" for b in chunk)
        # 16バイトに満たない場合はパディング
        hex_part = f"{hex_part:<48}"

        # 3. ASCII表現
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)

        print(f"{offset_str}  {hex_part}  |{ascii_part}|")


def display_chunk_data(filepath, chunk_name, offset, size):
    """
    指定されたチャンクのデータを読み込み、16進ダンプで表示する。
    """
    print(f"\n--- {chunk_name} (Offset: {offset}, Size: {size}) のデータ ---")

    # 表示する最大バイト数
    max_bytes_to_show = 256

    try:
        with open(filepath, "rb") as f:
            f.seek(offset)
            # チャンクサイズが256バイトより小さい場合も考慮
            bytes_to_read = min(size, max_bytes_to_show)
            data_to_show = f.read(bytes_to_read)

            hexdump(data_to_show, offset)

            if size > max_bytes_to_show:
                print(f"... (残り {size - max_bytes_to_show} バイトは省略されました)")

    except Exception as e:
        print(f"エラー: データの読み込みに失敗しました - {e}")

    print("--------------------------------------------------\n")


def parse_chunks(file, limit_offset, current_path, indent, chunk_map):
    """
    RIFFファイルを再帰的に解析し、チャンク構造を表示する。

    :param file: ファイルオブジェクト
    :param limit_offset: この関数が読み込むべき終端オフセット
    :param current_path: 現在のチャンクパス (例: "RIFF.LIST_pdta")
    :param indent: インデントレベル
    :param chunk_map: チャンクパスをキー、(offset, size)を値とする辞書
    """
    while file.tell() < limit_offset:
        try:
            # チャンクID (4バイト)
            chunk_id_bytes = file.read(4)
            if not chunk_id_bytes:
                break  # ファイル終端
            chunk_id = chunk_id_bytes.decode("ascii")

            # チャンクサイズ (4バイト, リトルエンディアン)
            chunk_size = int.from_bytes(file.read(4), "little")

        except Exception:
            # チャンクの途中でファイルが終わった場合など
            print("  " * indent + "(ファイルの終端または破損したチャンク)")
            break

        # データ本体の開始オフセット
        data_offset = file.tell()

        # チャンクデータの終端（パディング前）
        chunk_end = data_offset + chunk_size

        # RIFFは2バイトアライメント。サイズが奇数の場合、1バイトのパディングが入る
        padding = chunk_size % 2

        # フルパスの生成
        full_chunk_path = f"{current_path}.{chunk_id}" if current_path else chunk_id

        # 1. 階層表示
        print("  " * indent + f"- {chunk_id} (Size: {chunk_size} bytes, Data Offset: {data_offset})")

        # 2. チャンクマップへの登録
        chunk_map[full_chunk_path] = (data_offset, chunk_size)

        # "RIFF" または "LIST" チャンクは入れ子構造を持つ
        if chunk_id == "RIFF" or chunk_id == "LIST":
            # フォームタイプ (e.g., "sfbk", "pdta", "INFO")
            form_type = file.read(4).decode("ascii")
            print("  " * (indent + 1) + f"({form_type})")

            # このチャンクの終わり（chunk_end）をリミットとして再帰呼び出し
            parse_chunks(file, chunk_end, full_chunk_path, indent + 1, chunk_map)

        # 次のチャンクの開始位置にシーク
        # (パディングを含めたチャンクの終端)
        file.seek(chunk_end + padding)


def interactive_shell(filepath, chunk_map):
    """
    ユーザー入力を受け付け、チャンクデータを表示する対話型シェル。
    """
    print("--- チャンクデータビューア ---")
    print("表示したいチャンクのフルパスを入力してください。")
    print("例: RIFF.LIST_pdta.phdr")
    print("終了するには \"q\" または \"exit\" と入力してください。")

    while True:
        try:
            user_input = input("\nChunkPath> ").strip()

            if user_input.lower() in ["q", "exit"]:
                print("終了します。")
                break

            if user_input in chunk_map:
                offset, size = chunk_map[user_input]
                display_chunk_data(filepath, user_input, offset, size)
            else:
                print(f"エラー: チャンク \"{user_input}\" は見つかりません。")
                print("利用可能なチャンクパス:")
                for path in chunk_map.keys():
                    print(f"  {path}")

        except EOFError:
            print("\n終了します。")
            break
        except KeyboardInterrupt:
            print("\n終了します。")
            break


def main():
    if len(sys.argv) < 2:
        print(f"使用方法: python {sys.argv[0]} <soundfont_file.sf2>")
        sys.exit(1)

    filepath = sys.argv[1]

    if not os.path.exists(filepath):
        print(f"エラー: ファイルが見つかりません - {filepath}")
        sys.exit(1)

    # チャンクのパスと (オフセット, サイズ) をマッピングする辞書
    chunk_map = {}

    try:
        with open(filepath, "rb") as f:
            # ファイルの終端オフセットを取得
            f.seek(0, 2)
            file_size = f.tell()
            f.seek(0)

            print(f"--- {filepath} のチャンク構造 ---")
            # 解析開始
            parse_chunks(f, file_size, "", 0, chunk_map)
            print("--------------------------------------------------")

        # 対話型シェルを起動
        interactive_shell(filepath, chunk_map)

    except Exception as e:
        print(f"ファイルの解析中にエラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
