# LLM Finetune

实验细节看：[![swanlab](https://img.shields.io/badge/Qwen2%20指令微调-SwanLab-438440)](https://swanlab.cn/@ZeyiLin/Qwen2-fintune/runs/cfg5f8dzkp6vouxzaxlx6/chart)

## 准备工作

安装环境：`pip install -r requirements.txt`

数据集下载：在[huangjintao/zh_cls_fudan-news](https://modelscope.cn/datasets/huangjintao/zh_cls_fudan-news/files)下载`train.jsonl`和`test.jsonl`到根目录下。

## 训练

- qwen2-1.5b 指令微调：`python train_qwen2.py`，或跟随`train_qwen2.ipynb`
- glm4-9b 指令微调：`python train_glm4.py`
