import re
uk_text_file_path = 'uk_article_text.txt' 

category_names = []

# \[\[Category:(.+?)(?:\|.+)?\]\]
# \[でエスケープ処理
# (.+?) : これがカテゴリ名で一文字以上の任意文字を( ]] )手前まで最短でキャプチャ
# (?:\|.+)? : オプションのソートキー部分 (|* )
# EX:[[Category:England|*]]ならEnglandをキャプチャする
pattern = re.compile(r'\[\[Category:(.+?)(?:\|.+)?\]\]')
with open(uk_text_file_path, 'r', encoding='utf-8') as f:
    text = f.read()
# findall() はパターンにマッチした全てのキャプチャグループ（カテゴリ名）のリストを返す
category_names = pattern.findall(text)
if category_names:
    for name in category_names:
        print(name)
else:
    print("カテゴリ名が見つかりませんでした。")
print(f"抽出されたカテゴリ名の総数: **{len(category_names)}**")