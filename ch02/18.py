from collections import Counter


def analyze_name_frequency(filepath):
    """
    ファイルの1列目の文字列の出現頻度を求めて頻度の高い順に出力
    """
    first_column_data = []

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                columns = line.strip().split('\t')

                if columns:
                    first_column_data.append(columns[0])

        frequency_map = Counter(first_column_data)

        # list of (名前, 頻度) タプルのリストを，頻度 (x[1]) をキーに降順 (reverse=True) でソート
        sorted_frequencies = sorted(
            frequency_map.items(),
            key=lambda item: item[1],
            reverse=True
        )

        print("--- 名前と出現頻度 (頻度順) ---")
        for name, count in sorted_frequencies:
            print(f"{count}\t{name}")

    except FileNotFoundError:
        print(f"エラー: ファイル '{filepath}' が見つかりません。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


analyze_name_frequency('popular-names.txt')
