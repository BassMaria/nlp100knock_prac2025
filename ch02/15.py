import math

N = 10

with open('popular-names.txt', 'r', encoding='utf-8') as f:
    total_lines = sum(1 for line in f)

lines_per_file = math.ceil(total_lines / N)

print(f"総行数: {total_lines}")
print(f"1ファイルあたりの行数: {lines_per_file}")


def split_file_by_lines(input_filepath, n_parts, output_prefix='x'):
    """
    ファイルを n_parts に行単位で分割し、連番のファイルに格納します。
    """
    try:
        # 総行数を計算
        with open(input_filepath, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)

        if total_lines == 0:
            print("エラー: ファイルが空です。")
            return

        # 1ファイルあたりの行数を計算（切り上げ）
        lines_per_file = math.ceil(total_lines / n_parts)

        # ファイルの読み込みと分割
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            part_num = 1
            current_line_count = 0
            outfile = None

            for i, line in enumerate(infile):
                if i % lines_per_file == 0:
                    if outfile:
                        outfile.close()

                    padding = len(str(n_parts))
                    output_filename = f"{output_prefix}{part_num:0{padding}d}"
                    print(f"ファイル {output_filename} を作成中...")

                    outfile = open(output_filename, 'w', encoding='utf-8')
                    part_num += 1

                if outfile:
                    outfile.write(line)

            if outfile:
                outfile.close()

        print("ファイル分割が完了しました。")

    except FileNotFoundError:
        print(f"エラー: ファイル '{input_filepath}' が見つかりません。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


split_file_by_lines('popular-names.txt', N)
