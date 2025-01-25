# 子任务2：中药处方推荐（TCM Prescription Recommendation）

## 运行方式
1. 首先运行`data_process.py`将数据集中的训练集转换成大模型微调数据格式。
2. 使用微调框架进行LoRA微调，baseline使用ms-swift作为微调框架。
3. 下载`Qwen-2.5-7B-Instruct`模型。
4. 运行`sft.sh`训练脚本，进行LLM微调。
5. 微调完以后使用`deploy.sh`脚本部署LLM。
6. 运行`infer.py`调用部署模型生成验证集预测结果数据格式为`jsonl`。
7. 最后运行`jsonl2json.py`代码将预测结果规格化。
