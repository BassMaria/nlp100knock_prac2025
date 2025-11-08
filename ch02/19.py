def sort_by_third_column(filepath):
    """
    ファイルを読み込み、3列目の数値（人数）の降順で各行をソートして出力します。
    """
    data = []

    try:
        # 1. ファイルからすべての行を読み込み、3列目の値を数値に変換
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                # 行をタブ '\t' で分割
                columns = line.strip().split('\t')

                if len(columns) >= 3:
                    try:
                        # 3列目の値を数値（整数）に変換
                        count = int(columns[2])
                        # ソートキーとなる数値と元の行全体をタプルで保存
                        data.append((count, line))
                    except ValueError:
                        # 3列目が数値でない行はスキップまたは警告
                        # print(f"警告: 3列目が数値ではありません: {line.strip()}")
                        pass

        # 2. 3列目の数値 (item[0]) をキーとして、降順 (reverse=True) でソート
        sorted_data = sorted(data, key=lambda item: item[0], reverse=True)

        # 3. ソートされた元の行を出力
        print("--- 3列目の数値の降順にソートされた結果 ---")
        for count, line in sorted_data:
            print(line, end='')

    except FileNotFoundError:
        print(f"エラー: ファイル '{filepath}' が見つかりません。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


sort_by_third_column('popular-names.txt')
