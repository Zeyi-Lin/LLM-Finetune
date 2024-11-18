# 导入所需的库
from modelscope.msdatasets import MsDataset
import os
import pandas as pd

MAX_DATA_NUMBER = 500

# 检查目录是否已存在
if not os.path.exists('coco_2014_caption'):
    # 从modelscope下载COCO 2014图像描述数据集
    ds =  MsDataset.load('modelscope/coco_2014_caption', subset_name='coco_2014_caption', split='train')
    print(len(ds))
    # 设置处理的图片数量上限
    total = min(MAX_DATA_NUMBER, len(ds))

    # 创建保存图片的目录
    os.makedirs('coco_2014_caption', exist_ok=True)

    # 初始化存储图片路径和描述的列表
    image_paths = []
    captions = []

    for i in range(total):
        # 获取每个样本的信息
        item = ds[i]
        image_id = item['image_id']
        caption = item['caption']
        image = item['image']

        # 保存图片并记录路径
        image_path = os.path.abspath(f'coco_2014_caption/{image_id}.jpg')
        image.save(image_path)

        # 将路径和描述添加到列表中
        image_paths.append(image_path)
        captions.append(caption)

        # 每处理50张图片打印一次进度
        if (i + 1) % 50 == 0:
            print(f'Processing {i+1}/{total} images ({(i+1)/total*100:.1f}%)')

    # 将图片路径和描述保存为CSV文件
    df = pd.DataFrame({
        'image_path': image_paths,
        'caption': captions
    })

    # 将数据保存为CSV文件
    df.to_csv('./coco-2024-dataset.csv', index=False)

    print(f'数据处理完成，共处理了{total}张图片')

else:
    print('coco_2014_caption目录已存在,跳过数据处理步骤')