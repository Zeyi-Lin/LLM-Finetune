# LLM Finetune

实验细节看：[![swanlab](https://img.shields.io/badge/Qwen2%20指令微调-SwanLab-438440)](https://swanlab.cn/@ZeyiLin/Qwen2-fintune/runs/cfg5f8dzkp6vouxzaxlx6/chart) [![swanlab](https://img.shields.io/badge/GLM4%20指令微调-SwanLab-438440)](https://swanlab.cn/@ZeyiLin/GLM4-fintune/runs/eabll3xug8orsxzjy4yu4/chart)

## 准备工作

安装环境：`pip install -r requirements.txt`

文本分类任务-数据集下载：在[huangjintao/zh_cls_fudan-news](https://modelscope.cn/datasets/swift/zh_cls_fudan-news/files)下载`train.jsonl`和`test.jsonl`到根目录下。

命名实体识别任务-数据集下载：在[qgyd2021/chinese_ner_sft](https://huggingface.co/datasets/qgyd2021/chinese_ner_sft/tree/main/data)下载`ccfbdci.jsonl`到根目录下。

## 训练

| 模型       | 任务              | 运行命令                                                             | Notebook | 文章                                                         |
| ---------- | ----------------- | -------------------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Qwen2-1.5b | 指令微调-文本分类 | python train_qwen2.py | [Jupyter Notebook](notebook/train_qwen2.ipynb) | [知乎](https://zhuanlan.zhihu.com/p/702491999) |
| Qwen2-1.5b    | 指令微调-命名实体识别 | python train_qwen2_ner.py | [Jupyter Notebook](notebook/train_qwen2_ner.ipynb) | [知乎](https://zhuanlan.zhihu.com/p/704463319)   |
| GLM4-9b    | 指令微调-文本分类 | python train_glm4.py | [Jupyter Notebook](notebook/train_glm4.ipynb) | [知乎](https://zhuanlan.zhihu.com/p/702608991)  |
| GLM4-9b    | 指令微调-命名实体识别 | python train_glm4_ner.py | [Jupyter Notebook](notebook/train_glm4_ner.ipynb) | [知乎](https://zhuanlan.zhihu.com/p/704719982)  |


## 推理

Qwen2系列：

```bash
python predict_qwen2.py
```

GLM4系列：

```bash
python predict_glm4.py
```

## 引用工具

- [SwanLab](https://github.com/SwanHubX/SwanLab)：AI训练记录、分析与可视化工具