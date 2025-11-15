import pandas as pd

uk_texts = 'uk_article_text.txt'

with open(uk_texts, encoding='utf-8') as f:
    lines = f.readlines()

ans = list(filter(lambda x: "[Category:" in x, lines))
print(ans)
