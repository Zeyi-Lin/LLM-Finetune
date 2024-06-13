# LLM Finetune

实验细节看：[![swanlab](https://img.shields.io/badge/Qwen2%20指令微调-SwanLab-438440)](https://swanlab.cn/@ZeyiLin/Qwen2-fintune/runs/cfg5f8dzkp6vouxzaxlx6/chart) [![swanlab](https://img.shields.io/badge/GLM4%20指令微调-SwanLab-438440)](https://swanlab.cn/@ZeyiLin/GLM4-fintune/runs/eabll3xug8orsxzjy4yu4/chart)

## 准备工作

安装环境：`pip install -r requirements.txt`

数据集下载：在[huangjintao/zh_cls_fudan-news](https://modelscope.cn/datasets/huangjintao/zh_cls_fudan-news/files)下载`train.jsonl`和`test.jsonl`到根目录下。

## 训练

| 模型       | 任务              | 运行命令                                                             | 文章                                                         |
| ---------- | ----------------- | -------------------------------------------------------------------- | ------------------------------------------------------------ |
| Qwen2-1.5b | 指令微调-文本分类 | `python train_qwen2.py`，或跟随[Jupyter Notebook](train_qwen2.ipynb) | [Qwen2 指令微调实战](https://zhuanlan.zhihu.com/p/702491999) |
| GLM4-9b    | 指令微调-文本分类 | `python train_glm4.py`，，或跟随[Jupyter Notebook](train_glm4.ipynb) | [GLM4 指令微调实战](https://zhuanlan.zhihu.com/p/702608991)  |
