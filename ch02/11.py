import sys


def display_head(filename, n):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for i in range(n):
                line = f.readline()
                if not line:
                    break
                print(line.rstrip('\n'), end='\n')  # 末尾の改行を削除してから改めて改行を出力
    except FileNotFoundError:
        print(f"エラー: ファイル '{filename}' が見つかりません。", file=sys.stderr)
    except Exception as e:
        print(f"エラーが発生しました: {e}", file=sys.stderr)


if __name__ == "__main__":
    input_file = "popular-names.txt"
    print("Nを入力してください")
    N = int(input())
    display_head(input_file, N)
