import pandas as pd

def main():
    result_file = 'out/54.csv'
    gold_file = 'questions-words.txt'
    # 正解データ(gold)を読み込み，カテゴリ情報を付与する
    gold_data = []
    current_category = None
    
    with open(gold_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith(':'):
                current_category = line[2:].strip()
            else:
                parts = line.split()
                if len(parts) == 4:
                    gold_data.append({
                        'category': current_category,
                        'word1': parts[0],
                        'word2': parts[1],
                        'word3': parts[2],
                        'expected': parts[3]
                    })

    df_gold = pd.DataFrame(gold_data)

    df_result = pd.read_csv(result_file)

    # 結合してカテゴリ情報を紐付ける
    # word1, word2, word3, expected が一致する行を結合
    merged_df = pd.merge(df_result, df_gold, on=['word1', 'word2', 'word3', 'expected'], how='inner')

    if len(merged_df) == 0:
        print("No matching data found between result and gold file.")
        return

    # アナロジーの種類（semantic/syntactic）を判定
    # gramで始まるカテゴリはsyntactic，それ以外はsemantic
    merged_df['analogy_type'] = merged_df['category'].apply(
        lambda x: 'syntactic' if x.startswith('gram') else 'semantic'
    )

    # 正解率の計算
    # predictedとexpectedが一致していれば正解
    merged_df['is_correct'] = merged_df['predicted'] == merged_df['expected']

    print(f"=== Evaluation Results ({len(merged_df)} samples) ===")
    
    # タイプごとの正解率
    for analogy_type, group in merged_df.groupby('analogy_type'):
        accuracy = group['is_correct'].mean()
        count = len(group)
        print(f"{analogy_type.capitalize()} Analogy: {accuracy:.4f} (n={count})")

    # 全体の正解率
    overall_accuracy = merged_df['is_correct'].mean()
    print(f"Overall Accuracy: {overall_accuracy:.4f}")

if __name__ == "__main__":
    main()
