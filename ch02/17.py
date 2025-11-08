def count_unique_first_column(filepath):
    """
    ファイルの1列目に含まれるユニークな文字列の種類を数えます。
    """
    unique_names = set()  # 重複を許さない集合（Set）

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                # 行をタブ '\t' で分割
                columns = line.strip().split('\t')

                # 1列目（インデックス0）を集合に追加
                if columns:
                    unique_names.add(columns[0])

        # ユニークな文字列の種類（集合の要素数）を出力
        print(f"1列目の文字列の異なり（種類）の数: {len(unique_names)}")
        print("\n--- 異なりの一覧（一部） ---")
        # 異なりの一覧をアルファベット順に出力（見やすいように10個だけ）
        for name in sorted(list(unique_names))[:10]:
            print(name)

    except FileNotFoundError:
        print(f"エラー: ファイル '{filepath}' が見つかりません。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


count_unique_first_column('popular-names.txt')
