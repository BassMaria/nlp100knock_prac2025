import re
import pandas as pd
uk_text_file_path = 'uk_article_text.txt' 

with open(uk_text_file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# セクション行を抽出するためのパターン
# ^(={2,}) : 行頭から始まり，2個以上の等号が続くグループ1
# (.*?) : 最小限のマッチングで，セクション名（グループ2）
# \1$ : グループ1と同じ数の等号が行末にあることを確認
# re.MULTLINEは行の先頭^末尾$を独立に見るので行ごとに見れるようになってる
pattern = re.compile(r"^(={2,})\s*(.*?)\s*\1$", re.MULTILINE)
# finditerはマッチした部分をイテレータで返すもの
sections_data = []
for match in pattern.finditer(text):
    equals_signs = match.group(1)
    section_name = match.group(2).strip()
    
    # 等号の数からレベルを計算 (==ならレベル1, ===ならレベル2)
    level = len(equals_signs) - 1 
    
    sections_data.append([section_name, level])

df = pd.DataFrame(sections_data, columns=['セクション名', 'レベル'])

print(df)