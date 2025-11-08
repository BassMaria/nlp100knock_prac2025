def extract_first_column(filepath, n_lines=10):
    """
    ファイルの先頭 n_lines 行について、各行の1列目（タブ区切りの最初の要素）を抽出して出力します。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i >= n_lines:
                    break

                # 行をタブ '\t' で分割
                columns = line.strip().split('\t')

                if columns:
                    print(columns[0])

    except FileNotFoundError:
        print(f"エラー: ファイル '{filepath}' が見つかりません。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


extract_first_column('popular-names.txt', 10)
