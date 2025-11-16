import re
import pandas as pd
uk_text_file_path = 'uk_article_text.txt' 

with open(uk_text_file_path, 'r', encoding='utf-8') as f:
    text = f.read()

pattern = re.compile(r"\[\[(ファイル|File):(.*?)(?:\||\]\])",re.MULTILINE)
extracted_files = []
for match in pattern.finditer(text):
    file_name = match.group(2)
    extracted_files.append(file_name)

df = pd.DataFrame(extracted_files, columns=['ファイル名']) 
print(df)