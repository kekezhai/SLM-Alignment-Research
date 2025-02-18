from transformers import AutoModel, AutoTokenizer, AutoConfig,AutoModelForCausalLM
import torch

#加载两个 .safetensors 文件
model_path1 = './train_2024-10-16-07-38-58/checkpoint-100' #sft,5e-8
model_path2 = './train_2024-09-06-10-57-00/checkpoint-1500' #sft,1e-6

#加载模型配置（假设两个模型配置相同）
config1 = AutoConfig.from_pretrained(model_path1)
config2 = AutoConfig.from_pretrained(model_path2)

#加载模型权重
#model1 = AutoModel.from_pretrained(model_path1, config=config1)
#model2 = AutoModel.from_pretrained(model_path2, config=config2)
model1 = AutoModelForCausalLM.from_pretrained(model_path1, config=config1)
model2 = AutoModelForCausalLM.from_pretrained(model_path2, config=config2)

#初始化一个新的模型实例，用于存储平均权重
#averaged_model = AutoModel.from_config(config1)
merged_model = AutoModelForCausalLM.from_config(config1)

#计算权重平均值
for name_param1,name_param2 in zip(model1.named_parameters(),model2.named_parameters()):
    name1,param1 = name_param1
    name2,param2 = name_param2
    #averaged_param = (param1 + param2) / 2
    merged_param = param1*0.05 + param2*0.95
    merged_model.state_dict()[name1].copy_(merged_param)

#保存平均权重
output_dir = './sft_checkpoint_merged_model2'
merged_model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(model_path1)  # 假设分词器相同
tokenizer.save_pretrained(output_dir)
print(f"merged model saved to {output_dir}")
