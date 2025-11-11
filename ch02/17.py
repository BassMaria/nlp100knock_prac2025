def count_unique_first_column(filepath):
    """
    ファイルの1列目に含まれるユニークな文字列の種類を数える
    """
    unique_names = set()  # 重複を許さない集合（Set）

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                columns = line.strip().split('\t')

                if columns:
                    unique_names.add(columns[0])

        print(f"1列目の文字列の異なり（種類）の数: {len(unique_names)}")
        print("\n--- 異なりの一覧（一部） ---")
        for name in sorted(list(unique_names))[:10]:
            print(name)

    except FileNotFoundError:
        print(f"エラー: ファイル '{filepath}' が見つかりません．")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


count_unique_first_column('popular-names.txt')
