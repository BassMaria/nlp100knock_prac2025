def extract_first_column(filepath, n_lines=10):
    """
    ファイルの先頭 n_lines 行について各行の1列目（タブ区切りの最初の要素）を抽出して出力
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                # 10行見たらループ終了
                if i >= n_lines:
                    break

                # 各行の先頭末尾の空白文字(スペース，タブ，改行など)除去後
                # タブ '\t' で分割
                columns = line.strip().split('\t')

                if columns:
                    print(columns[0])

    except FileNotFoundError:
        print(f"エラー: ファイル '{filepath}' が見つかりません．")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


# defの引数でn_lines=10としているのであえて明示しない場合10行表示してくれる
extract_first_column('popular-names.txt')
