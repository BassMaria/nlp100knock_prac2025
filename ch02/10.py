import sys


def count_file_lines(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            # f.readlines()で全ての行をリストとして読み込み，その長さを取得
            line_count = len(f.readlines())
        print(line_count)
    except FileNotFoundError:
        print(f"エラー: ファイル '{filename}' が見つかりません", file=sys.stderr)
    except Exception as e:
        print(f"エラーが発生しました: {e}", file=sys.stderr)


if __name__ == "__main__":
    input_file = "popular-names.txt"
    count_file_lines(input_file)
