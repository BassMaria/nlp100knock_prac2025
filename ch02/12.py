import sys
from collections import deque


def display_tail(filename, n):
    try:
        # dequeを使うと，最大長Nを超えた場合に古い要素を自動で削除しながら行を格納できて
        # メモリ効率良くファイルの最後のN行を保持できる．
        last_n_lines = deque(maxlen=n)

        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                last_n_lines.append(line.rstrip('\n'))

        for line in last_n_lines:
            print(line)

    except FileNotFoundError:
        print(f"エラー: ファイル '{filename}' が見つかりません", file=sys.stderr)
    except Exception as e:
        print(f"エラーが発生しました: {e}", file=sys.stderr)


if __name__ == "__main__":
    input_file = "popular-names.txt"
    print("Nを入力してください")
    N = int(input())
    display_tail(input_file, N)
