from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
mode_path = '/root/autodl-tmp/qwen/Qwen2-7B-Instruct/'
lora_path = './output/Qwen2_instruct_lora/checkpoint-10' # 这里改称你的 lora 输出对应 checkpoint 地址``
# 加载tokenizer
# tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)
# # 加载模型
# model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
# # 加载lora权重
# model = PeftModel.from_pretrained(model, model_id=lora_path)
