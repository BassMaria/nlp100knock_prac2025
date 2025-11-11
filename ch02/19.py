def sort_by_third_column(filepath):
    """
    ファイルを読み込み3列目の数値（人数）の降順で各行をソートして出力
    """
    data = []

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                columns = line.strip().split('\t')

                if len(columns) >= 3:
                    try:
                        count = int(columns[2])
                        data.append((count, line))
                    except ValueError:
                        pass

        # 3列目の数値 (item[0]) をキーとして，降順 (reverse=True) でソート
        sorted_data = sorted(data, key=lambda item: item[0], reverse=True)

        print("--- 3列目の数値の降順にソートされた結果 ---")
        for count, line in sorted_data:
            print(line, end='')

    except FileNotFoundError:
        print(f"エラー: ファイル '{filepath}' が見つかりません．")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


sort_by_third_column('popular-names.txt')
