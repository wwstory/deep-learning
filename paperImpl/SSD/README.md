# introduce
- 对`ssd.pytorch`项目有了一些改动，改了部分过时的方法。
- 改为COCO数据集的训练与测试。
- 新增一些测试。
- 加了一些方法可视化。


# install

`requirements.txt`
```
torch==1.4.0
torchvision==0.5.0
Cython==0.29.14
pycocotools==2.0
opencv-python==4.1.2.30
matplotlib==3.1.2
Pillow==6.2.1
```

# test
`test/`目录对一些函数有测试。

# 目录结构
```
.
├── checkpoints/
│   └── COCO.pth
├── config.py
├── data/
│   ├── augmentations.py
│   ├── coco.py
│   ├── __init__.py
│   └── scripts/
│       ├── COCO2014.sh
│       ├── VOC2007.sh
│       └── VOC2012.sh
├── detect.py   # 封装好的ssd用于预测的类
├── layers/
│   ├── box_utils.py
│   ├── functions/
│   │   ├── detection.py
│   │   ├── __init__.py
│   │   └── prior_box.py
│   ├── __init__.py
│   └── modules/
│       ├── __init__.py
│       ├── l2norm.py
│       └── multibox_loss.py
├── out/
├── README.md
├── ssd.py
├── test/   # 测试函数
│   ├── all_classes.txt
│   ├── _augmentations.py
│   ├── coco_labels_map.txt
│   ├── test_augmentations.py
│   ├── test_box_utils.py
│   ├── test_coco.py
│   ├── test_detection.py
│   ├── test_detect.py
│   ├── test_multibox_loss.py
│   ├── test_prior_box.py
│   ├── test_pycoco.py
│   ├── test_ssd.py
│   ├── test_test_specify_classes.py
│   └── test_train.py
├── test.py     # 从valid验证集中抽选出数据测试
└── train.py    # 用于训练COCO数据集
```

# dataset
下载训练数据集脚本：`data/scripts/COCO2014.sh`

