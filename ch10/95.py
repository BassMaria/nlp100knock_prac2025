from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

# --- 初期チャット（問題94の状態） ---
chat = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Please answer the following questions.",
    },
    {"role": "user", "content": "What do you call a sweet eaten after dinner?"},
]

# プロンプト生成
prompt = tokenizer.apply_chat_template(chat, tokenize=False)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 1回目の応答生成
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

# 生成部分のみ取り出す
generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
answer1 = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("1回目の応答:")
print(answer1)

# --- 応答を履歴に追加 ---
chat.append({"role": "assistant", "content": answer1})

# --- 追加質問（問題95） ---
chat.append({
    "role": "user",
    "content": "Please give me the plural form of the word with its spelling in reverse order."
})

# プロンプト確認
prompt2 = tokenizer.apply_chat_template(chat, tokenize=False)

print("\nモデルに与えるプロンプト:")
print(prompt2)

# トークン化
inputs2 = tokenizer(prompt2, return_tensors="pt").to(model.device)

# 応答生成
outputs2 = model.generate(**inputs2, max_new_tokens=50,do_sample=False)

generated_tokens2 = outputs2[0][inputs2.input_ids.shape[1]:]
answer2 = tokenizer.decode(generated_tokens2, skip_special_tokens=True)

print("\n2回目の応答:")
print(answer2)
