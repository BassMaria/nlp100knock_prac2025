from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# モデルとトークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# チャットの構造
chat = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Please answer the following questions.",
    },
    {"role": "user", "content": "What do you call a sweet eaten after dinner?"},
    {"role": "assistant", "content": "A sweet eaten after dinner is called a dessert."},
    {
        "role": "user",
        "content": "Please give me the plural form of the word with its spelling in reverse order.",
    },
]

# チャットテンプレートを適用
prompt = tokenizer.apply_chat_template(chat, tokenize=False)
print("生成されたプロンプト:")
print(prompt)
print()

# トークン化
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 応答生成
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    repetition_penalty=1.1,
)

# デコード
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("生成された応答:")
print(response)
