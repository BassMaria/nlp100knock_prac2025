import re
import pandas as pd
uk_text_file_path = 'uk_article_text.txt' 

with open(uk_text_file_path, 'r', encoding='utf-8') as f:
    text = f.read()
template_start = text.find('{{基礎情報 国')
start_index = template_start + len('{{基礎情報 国')
# ネストされたテンプレートに対応するため括弧のカウントで終了位置を探す
count = 1
end_index = -1
for i in range(start_index, len(text)):
    if text[i:i+2] == '{{':
        count += 1
    elif text[i:i+2] == '}}':
        count -= 1
        if count == 0:
            end_index = i
            break

# 終了位置が正しく見つかった場合テンプレートブロックを抽出
if end_index != -1:
    template_block = text[start_index:end_index].strip()
else:
    # 見つからない場合エラーを避けるために空文字列にするか元の簡易抽出を試みる
    template_block = ""

pattern = re.compile(r'^\s*\|(.+?)\s*=\s*(.+?)(?=\n[|}])', re.MULTILINE | re.DOTALL)
# re.DOTALL: . が改行にもマッチするようにする（値が複数行にわたる場合に対応）
field_dict = {}
for match in pattern.finditer(template_block):
    field_name = match.group(1).strip()
    value = match.group(2).strip()
    # 25に以下追記
    value = re.sub(r"'''''","",value)
    value = re.sub(r"'''","",value)
    value = re.sub(r"''","",value)
    field_dict[field_name] = value

print(field_dict)