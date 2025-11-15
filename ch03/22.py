import re
uk_text_file_path = 'uk_article_text.txt' 

category_names = []

# 正規表現パターン: 
# \[\[Category:(.+?)(?:\|.+)?\]\]
# (.+?) : これがカテゴリ名（非貪欲マッチで最短一致）
# (?:\|.+)? : オプションのソートキー部分 (|* など)
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