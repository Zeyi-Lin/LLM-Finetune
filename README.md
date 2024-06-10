# LLM Finetune

## 准备工作

安装环境：`pip install -r requirements.txt`

数据集下载：在[huangjintao/zh_cls_fudan-news](https://modelscope.cn/datasets/huangjintao/zh_cls_fudan-news/files)下载`train.jsonl`和`test.jsonl`到根目录下。

## 训练

- qwen2-1.5b指令微调：`python train_qwen2.py`
- glm4-9b指令微调：`python train_glm4.py`