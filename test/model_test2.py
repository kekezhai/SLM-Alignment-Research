from transformers import pipeline
import torch

# 将模型移动到 GPU（如果有的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id =  "./Qwen2-0.5B-Instruct-S1"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map=device,
)

def chat(messages):
    outputs = pipe(
        messages,
        max_new_tokens=1024,
    )
    return outputs[0]["generated_text"][-1]['content']

if __name__ == "__main__":
    prompt = "怎么学习大模型"
    messages = [{"role": "user", "content": prompt}]
    answer = chat(messages)
    print(answer)
