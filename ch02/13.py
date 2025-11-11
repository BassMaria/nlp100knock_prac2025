def replace_tabs_with_spaces(filepath, n_lines=10):
    """
    ファイルの先頭 n_lines 行についてタブをスペースに置換して出力
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i >= n_lines:
                    break
                modified_line = line.replace('\t', ' ')
                print(modified_line.strip('\n'))

    except FileNotFoundError:
        print(f"エラー: ファイル '{filepath}' が見つかりません．")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


# defの引数でn_lines=10としているのであえて明示しない場合10行表示してくれる
replace_tabs_with_spaces('popular-names.txt')
