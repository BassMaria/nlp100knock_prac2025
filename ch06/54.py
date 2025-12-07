from gensim.models import KeyedVectors
import pandas as pd
from tqdm import tqdm

# 学習済み単語ベクトル path
model_path = 'GoogleNews-vectors-negative300.bin.gz'

print("Loading model...")
# バイナリ形式のモデルを読み込む
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

eval_file_path = 'questions-words.txt'
output_file = 'out/54.csv'

results = []
target_section = ": capital-common-countries"
in_target_section = False

print(f"Processing {target_section} from {eval_file_path}...")

try:
    with open(eval_file_path, 'r') as f:
        for line in tqdm(f):
            line = line.strip()
            # セクションの判定
            if line.startswith(':'):
                if line == target_section:
                    in_target_section = True
                else:
                    in_target_section = False
                continue
            
            if not in_target_section:
                continue
                
            # フォーマット: word1 word2 word3 word4
            # 計算: vec(word2) - vec(word1) + vec(word3)
            parts = line.split()
            if len(parts) != 4:
                continue
                
            w1, w2, w3, w4 = parts
            
            try:
                # positive=[w2, w3], negative=[w1] で vec(w2) - vec(w1) + vec(w3) の計算を行って，類似度が高い単語を探す
                most_similar = model.most_similar(positive=[w2, w3], negative=[w1], topn=1)[0]
                pred_word = most_similar[0]
                similarity = most_similar[1]
                
                results.append({
                    'word1': w1,
                    'word2': w2,
                    'word3': w3,
                    'expected': w4,
                    'predicted': pred_word,
                    'similarity': similarity
                })
            except KeyError:
                # 語彙にない単語が含まれる場合はスキップ
                continue

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    print(df.head())

except FileNotFoundError:
    print(f"Error: {eval_file_path} がないので先にダウンロードしてください")

