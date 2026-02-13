import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm


# 設定
model_dir = "./ch10/results/gpt2-dpo"
dev_path = "./ch07/SST-2/dev.tsv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# モデル読み込み
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

tokenizer.pad_token = tokenizer.eos_token
model.to(device)
model.eval()


# データ読み込み


dev_df = pd.read_csv(dev_path, sep="\t")


# 予測関数
def predict_sentiment(text):
    prompt = f"""You are a sentiment classifier.

Sentence: {text}
Sentiment:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False
        )

    # 生成部分のみ取得
    input_len = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_len:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).lower().strip()

    if "positive" in response:
        return 1
    elif "negative" in response:
        return 0
    else:
        return -1  # 判定不能



# 評価


correct = 0
total = 0
invalid = 0

for _, row in tqdm(dev_df.iterrows(), total=len(dev_df)):
    text = row["sentence"]
    label = row["label"]

    pred = predict_sentiment(text)

    if pred == -1:
        invalid += 1
    elif pred == label:
        correct += 1

    total += 1

accuracy = correct / total

print("\n===== DEV RESULT =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Correct  : {correct}")
print(f"Total    : {total}")
print(f"Invalid  : {invalid}")
