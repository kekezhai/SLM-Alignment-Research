#from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 将模型移动到 GPU（如果有的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = './Qwen2-0.5B-Instruct-S1'

tokenizer = AutoTokenizer.from_pretrained(model_path)
#使用半精度half().cuda()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device
).half().cuda()

model.to(device)

#def chat(messages, max_new_tokens=1024,temperature=0.7,top_p=0.7):
def chat(messages, max_new_tokens=1024,temperature=0.95,top_p=0.7):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens= max_new_tokens,
        temperature = temperature,
        top_p = top_p
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

if __name__ == '__main__':
    prompt = "怎么学习大模型"
    messages = [{"role": "user", "content": prompt}]
    answer = chat(messages)
    print(answer)
