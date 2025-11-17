import re
import requests
import json

uk_text_file_path = 'uk_article_text.txt'

def remove_stress(dc):
    r = re.compile("'+")
    return {k: r.sub("", v) for k, v in dc.items()}

def remove_inner_links(dc):
    r = re.compile(r"\[\[(?:[^|]*?\|)?([^|]+?)\]\]")
    return {k: r.sub(r"\1", v) for k, v in dc.items()}

def remove_templates_safely(dc):
    """{{...}} から表示部分だけ抽出し、完全削除はしない"""
    r = re.compile(r"\{\{(?:[^|]*?\|)?([^|}]+?)\}\}")
    return {k: r.sub(r"\1", v) for k, v in dc.items()}

def remove_br(dc):
    r = re.compile(r"<\s*?/*?\s*?br\s*?/*?\s*>", re.IGNORECASE)
    return {k: r.sub("", v) for k, v in dc.items()}


# ---- ファイル読み込み ----
with open(uk_text_file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# ---- 基礎情報ブロック抽出 ----
template_start = text.find('{{基礎情報 国')
start_index = template_start + len('{{基礎情報 国')
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

template_block = text[start_index:end_index].strip()

# ---- key=value 抽出 ----
pattern = re.compile(r'^\s*\|(.+?)\s*=\s*(.+?)(?=\n[|}])', re.MULTILINE | re.DOTALL)
field_dict = {}

for match in pattern.finditer(template_block):
    key = match.group(1).strip()
    value = match.group(2).strip()
    field_dict[key] = value


# ---- マークアップ除去処理 ----
cleaned = remove_stress(field_dict)
cleaned = remove_inner_links(cleaned)
cleaned = remove_templates_safely(cleaned)
cleaned = remove_br(cleaned)

# ---- 国旗画像取得 ----
filename = cleaned['国旗画像'].replace(' ', '_')

URL = "https://ja.wikipedia.org/w/api.php"
PARAMS = {
    "action": "query",
    "titles": "File:" + filename,
    "prop": "imageinfo",
    "iiprop": "url",
    "format": "json"
}
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NLP100-Practice/1.0)"}

res = requests.get(URL, params=PARAMS, headers=HEADERS)

# JSON パース確認
data = res.json()
page = data['query']['pages']
page_id = list(page.keys())[0]
image_url = page[page_id]['imageinfo'][0]['url']

print(image_url)
