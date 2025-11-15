import pandas as pd

output_file_name = 'uk_article_text.txt'
df = pd.read_json("jawiki-country.json.gz", lines=True)
uk_text = df.query('title=="イギリス"')["text"].values[0]
with open(output_file_name, 'w', encoding='utf-8') as out_f:
        out_f.write(uk_text)
print("記事本文を抽出成功")