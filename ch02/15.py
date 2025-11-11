import math

N = 10
# デバッグのための処理

with open('popular-names.txt', 'r', encoding='utf-8') as f:
    # lineはファイル内の1行1行でlineの数を数えている
    total_lines = sum(1 for line in f)
# math.ceil()は切り上げができる関数
# 例：101行をファイル10個に分けたいとき math.ceil(101/10) = 11 で漏れない
lines_per_file = math.ceil(total_lines / N)

print(f"総行数: {total_lines}")
print(f"1ファイルあたりの行数: {lines_per_file}")


def split_file_by_lines(input_filepath, n_parts, output_prefix='x'):
    """
    ファイルを n_parts に行単位で分割して連番のファイルに格納
    """
    try:
        # ファイル読み込みと全行数カウント
        with open(input_filepath, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)

        if total_lines == 0:
            print("エラー: ファイルが空です．")
            return

        lines_per_file = math.ceil(total_lines / n_parts)

        # 再度ファイルの読み込み(ファイルポインタが末尾のため)と分割
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            part_num = 1
            outfile = None

            for i, line in enumerate(infile):
                if i % lines_per_file == 0:
                    # 新しい分割用ファイルを開くため古いファイルをclose
                    if outfile:
                        outfile.close()
                    # x0(padding)dで分割ファイル連番名をゼロ埋めでソートしやすくする
                    padding = len(str(n_parts))
                    # dは整数の意味
                    output_filename = f"{output_prefix}{part_num:0{padding}d}"
                    print(f"ファイル {output_filename} を作成中...")

                    outfile = open(output_filename, 'w', encoding='utf-8')
                    part_num += 1

                if outfile:
                    outfile.write(line)

            if outfile:
                outfile.close()

        print("ファイル分割が完了しました．")

    except FileNotFoundError:
        print(f"エラー: ファイル '{input_filepath}' が見つかりません．")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


split_file_by_lines('popular-names.txt', N)
