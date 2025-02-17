# SLM-Alignment-Research

## 介绍
- **基于小型语言模型的后训练持续优化方法**

本文主要探索了小型语言模型的后训练持续优化方法，提出了一种针对小型语言模型的持续后训练对齐数据构建方法。该方法的核心是基于大模型的数据指导，优化了对齐数据的多样性和准确率，同时兼顾了模型的生成安全性。此外，为了验证本文方法的有效性，我们使用Qwen2-0.5B-Instruction作为SLM底座模型，并使用该方法构建的数据集进行了SFT后训练实验和KTO后训练实验，以及SFT-KTO两阶段实验和模型权重融合实验。最后，我们使用benchmark测试集对后训练模型进行了评估及分析，验证了持续后训练对于提升小型语言模型性能的效果。详见论文: [A Post-Training Enhanced Optimization Approach for Small Language Models](https://arxiv.org/abs/2411.02939)

## 模型地址
[Qwen2-0.5B-Instruct-S1](https://www.modelscope.cn/models/kkzhai/Qwen2-0.5B-Instruct-S1)

### 模型训练
git clone https://github.com/hiyouga/LLaMA-Factory.git
sh train.sh

### 模型评估
pip install -r ./eval/requirements.txt

python ./eval/test_evaluate_chat_gsm8k.py --use-fewshot -f ./Evaluation_DataSets/gsm8k -c Qwen2-0.5B-Instruct-S1模型目录  -o gsm8k_4shot_Qwen2-0.5B-Instruct-S1_res.jsonl 

python ./eval/test_evaluate_chat_mmlu.py -d ./Evaluation_DataSets/mmlu/data -c Qwen2-0.5B-Instruct-S1模型目录  

python ./eval/test_evaluate_chat_cmmlu.py -d ./Evaluation_DataSets/cmmlu/data -c Qwen2-0.5B-Instruct-S1模型目录  

python ./eval/test_evaluate_chat_ceval.py -d ./Evaluation_DataSets/ceval/data -c Qwen2-0.5B-Instruct-S1模型目录  

### HumanEval安装及评估
git clone https://github.com/openai/human-eval ./Evaluation_DataSets/human-eval
pip install -e ./Evaluation_DataSets/human-eval

python ./eval/test_evaluate_chat_humaneval.py -f ./Evaluation_DataSets/HumanEval/raw/human-eval/data/HumanEval.jsonl -c Qwen2-0.5B-Instruct-S1模型目录 -o ./HumanEval_Qwen2-0.5B-Instruct-S1_res.jsonl

evaluate_functional_correctness ./HumanEval_Qwen2-0.5B-Instruct-S1_res.jsonl
