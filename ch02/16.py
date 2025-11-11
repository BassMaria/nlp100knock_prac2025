import random


def shuffle_lines(filepath):
    """
    ファイルを読み込み，行単位でランダムに並び替えて出力
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        random.shuffle(lines)

        for line in lines:
            # readlines()で読み込んだ行には改行文字が含まれているためそのままprint
            print(line, end='')

    except FileNotFoundError:
        print(f"エラー: ファイル '{filepath}' が見つかりません．")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


shuffle_lines('popular-names.txt')
