import re

uk_text_file_path = 'uk_article_text.txt' 

def remove_stress(dc):
    """辞書の値からMediaWikiの強調マークアップ（'''、''）を除去する"""
    # 正規表現: 'が1回以上続くパターンにマッチ
    r = re.compile("'+")
    return {k: r.sub("", v) for k, v in dc.items()}

def remove_inner_links(dc):
    """辞書の値からMediaWikiの内部リンク（[[...]]）を除去し、表示テキストのみを残す"""
    # 正規表現: \[\[(?:[^|]*?\|)?([^|]+?)\]\] を \1 で置換
    r = re.compile(r"\[\[(?:[^|]*?\|)?([^|]+?)\]\]")
    return {k: r.sub(r"\1", v) for k, v in dc.items()}



with open(uk_text_file_path, 'r', encoding='utf-8') as f:
    text = f.read()

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

template_block = text[start_index:end_index].strip() if end_index != -1 else ""

# re.DOTALL: . が改行にもマッチするようにする（値が複数行にわたる場合に対応）
pattern = re.compile(r'^\s*\|(.+?)\s*=\s*(.+?)(?=\n[|}])', re.MULTILINE | re.DOTALL)
field_dict = {}

for match in pattern.finditer(template_block):
    field_name = match.group(1).strip()
    value = match.group(2).strip()
    field_dict[field_name] = value


# 辞書全体に対して強調マークアップ除去
cleaned_dict = remove_stress(field_dict)
# 内部リンク除去を適用
final_dict = remove_inner_links(cleaned_dict)

print(final_dict)