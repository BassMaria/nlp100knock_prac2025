from transformers import AutoTokenizer
import os

# 学習に使った tokenizer と同じもの
model_name = "bert-base-uncased"

# Trainer の output_dir
save_dir = "ch09/results/87/checkpoint-8420"

# 念のためディレクトリ確認
if not os.path.exists(save_dir):
    raise FileNotFoundError(f"{save_dir} does not exist")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_dir)

print("Tokenizer saved to:", save_dir)
