import google.genai as genai
import os
import pandas as pd
from tqdm import tqdm
import time
import re

api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)
# 今回は熟語を使用した
csv_file_path = 'japanese_idiom.csv'


# プロンプトを変更して実験
def get_model_answer(question, option_a, option_b, option_c, option_d):
    prompt = f"""
以下の問題に対する正解の選択肢を選び，その記号（A, B, C, D）を回答してください．
余計な装飾は一切不要ですが解答の根拠は示してください．

問題: {question}

選択肢:
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}

回答:
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return ""

def extract_choice(text):
    # 大文字に変換してA-Dを探す
    match = re.search(r'[ABCD]', text.upper())
    if match:
        return match.group(0)
    return None

def main():
    if not os.path.exists(csv_file_path):
        print(f"エラー: {csv_file_path} が見つかりません．")
        return

    # ヘッダーなしとして読み込みカラム名を付与
    df = pd.read_csv(csv_file_path, header=None, names=['問題', '選択肢A', '選択肢B', '選択肢C', '選択肢D', '正解'])
    # 5問に限定
    df = df.head(5)
    
    correct_count = 0
    total_count = len(df)
    results = []

    print(f"全{total_count}問の処理を開始します!")

    # tqdmで進捗バーを表示しながらループ
    for index, row in tqdm(df.iterrows(), total=total_count):
        raw_answer = get_model_answer(
            row['問題'], 
            row['選択肢A'], 
            row['選択肢B'], 
            row['選択肢C'], 
            row['選択肢D']
        )
        
        # 回答から記号を抽出
        model_choice = extract_choice(raw_answer)
        correct_choice = str(row['正解']).strip().upper()
        
        # 正解判定
        is_correct = (model_choice == correct_choice)
        if is_correct:
            correct_count += 1
            
        results.append({
            '問題': row['問題'],
            '正解': correct_choice,
            'モデル回答(生)': raw_answer,
            'モデル選択': model_choice,
            '判定': '正解' if is_correct else '不正解'
        })
        
        # APIレート制限対策（必要に応じて調整）
        time.sleep(0.2)

    # 結果表示
    accuracy = correct_count / total_count
    # 結果をリスト表示
    print(results)
    print(f"\n=== 結果 ===")
    print(f"正解数: {correct_count} / {total_count}")
    print(f"正解率: {accuracy:.2%}")

if __name__ == "__main__":
    main()