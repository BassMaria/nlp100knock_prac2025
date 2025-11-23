import re
import spacy
from collections import Counter
import pandas as pd
from tqdm import tqdm

open_file = "jawiki-country.json.gz"
nlp = spacy.load('ja_ginza')

# 正規表現パターンを事前にコンパイル
patterns = [
    r"\'{2,5}",  # 強調マークアップ
    r"\[\[(?:[^|\]]*?\|)?([^|\]]+?)\]\]",  # 内部リンク
    r"\[(https?://[^ ]+)( [^\]]+)?\]",  # 外部リンク
    r"<[^>]+>",  # HTMLタグ
    r"\{\{.*?\}\}",  # テンプレート
]
compiled_pattern = re.compile('|'.join(patterns), re.MULTILINE)

def remove_markup(text):
    old_text = ""
    while old_text != text:
        old_text = text
        text = compiled_pattern.sub(lambda m: m.group(1) if m.group(1) else '', text)
    return text

def split_text_by_bytes(text, max_bytes=49149):
    """Sudachiの制限に合わせてテキストを分割しリストで返す"""
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return [text]
    
    chunks = []
    for i in range(0, len(encoded), max_bytes):
        chunk = encoded[i:i + max_bytes]
        # 分割位置による文字化けは無視
        chunks.append(chunk.decode('utf-8', errors='ignore'))
    return chunks

def main():
    df = pd.read_json(open_file, orient='records', lines=True, compression='gzip')
    df['cleaned_text'] = df['text'].apply(remove_markup)
    # split_text_by_bytes はリストを返すので、explodeで展開して行を増やす
    df['chunks'] = df['cleaned_text'].apply(split_text_by_bytes)
    
    # explode: リストの要素を行に展開（これでforループのネストを回避）
    text_series = df.explode('chunks')['chunks']
    # 空の文字列を除外
    text_series = text_series[text_series.str.len() > 0]

    target_pos = {'NOUN', 'VERB', 'ADJ', 'ADV'} # setにして検索を高速化
    all_words = []
    
    tqdm_text_series = tqdm(
        text_series, 
        desc="形態素解析中 (NLP Pipe)", 
        total=len(text_series)
    )

    # nlp.pipe はイテレータを返す
    # disable=['ner', 'parser']: 単語抽出には不要な重い処理を無効化
    docs = nlp.pipe(tqdm_text_series, batch_size=200, n_process=4, disable=['ner', 'parser'])

    for doc in docs:
        for token in doc:
            if token.pos_ in target_pos:
                all_words.append(token.text)

    word_counts = Counter(all_words)
    top_20 = word_counts.most_common(20)

    print("単語\t出現頻度")
    for word, count in top_20:
        print(f"{word}\t{count}")

if __name__ == "__main__":
    main()