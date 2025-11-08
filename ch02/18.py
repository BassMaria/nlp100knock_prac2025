from collections import Counter


def analyze_name_frequency(filepath):
    """
    ファイルの1列目の文字列の出現頻度を求め、頻度の高い順に出力します。
    """
    first_column_data = []

    try:
        # 1. 1列目のデータをすべて抽出
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                # 行をタブ '\t' で分割
                columns = line.strip().split('\t')

                # 1列目（インデックス0）をリストに追加
                if columns:
                    first_column_data.append(columns[0])

        # 2. 出現頻度をカウント
        # { '名前': 頻度, ... } の辞書を作成
        frequency_map = Counter(first_column_data)

        # 3. 頻度でソート（降順）
        # list of (名前, 頻度) タプルのリストを、頻度 (x[1]) をキーに降順 (reverse=True) でソート
        sorted_frequencies = sorted(
            frequency_map.items(),
            key=lambda item: item[1],
            reverse=True
        )

        # 4. 結果を出力
        print("--- 名前と出現頻度 (頻度順) ---")
        for name, count in sorted_frequencies:
            # 「頻度 [タブ] 名前」の形式で出力 (見やすいようにタブ区切り)
            print(f"{count}\t{name}")

    except FileNotFoundError:
        print(f"エラー: ファイル '{filepath}' が見つかりません。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


analyze_name_frequency('popular-names.txt')
