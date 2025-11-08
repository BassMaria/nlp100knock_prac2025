import random


def shuffle_lines(filepath):
    """
    ファイルを読み込み、行単位でランダムに並び替えて出力します。
    """
    try:
        # 1. ファイルからすべての行を読み込む
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 2. 行のリストをランダムにシャッフルする
        random.shuffle(lines)

        # 3. シャッフルされた行を出力する
        for line in lines:
            # readlines()で読み込んだ行には改行文字が含まれているため、そのままprintする
            print(line, end='')

    except FileNotFoundError:
        print(f"エラー: ファイル '{filepath}' が見つかりません。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


shuffle_lines('popular-names.txt')
