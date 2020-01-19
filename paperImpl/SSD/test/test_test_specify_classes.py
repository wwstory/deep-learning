import os
import sys
os.chdir('..')
sys.path.append('.')
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# net
from ssd import build_ssd
# data
from data import COCO_CLASSES as CLASSES
from data import COCODateset, BaseTransform
from data.augmentations import SSDAugmentation
# other
from config import opt
from data.coco import COCOAnnotationTransform, coco_conf as cfg, MEANS
from layers.box_utils import select_data

from pycocotools.coco import COCO



if opt.use_gpu and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(opt.out_folder):
    os.mkdir(opt.out_folder)

# plot setting
import matplotlib
# all params: ~/venv/torch_venv/lib/python3.6/site-packages/matplotlib/mpl-data/matplotlibrc
matplotlib.rcParams['patch.edgecolor'] = '#ff0000'
matplotlib.rcParams['patch.linewidth'] = 1
# matplotlib.rcParams['font.family'] = 'YouYuan'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['text.color'] = '#0000ff'


# 设置类别和数量
IMAGE_SET = 'val2014'
NUM = 10
VALID_CLASSES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
    'chair', 'couch', 'potted plant', 'bed', 
)
# 生成数据索引
coco = COCO(f'{opt.root_path}annotations/instances_{IMAGE_SET}.json')
cat_ids = coco.getCatIds(VALID_CLASSES)
img_ids = []
for c in cat_ids:
    img_ids += coco.getImgIds(catIds=c)
ids = list(coco.imgToAnns.keys())
target_transform = COCOAnnotationTransform()
NUM = len(ids)


def test_net(save_folder, net, use_gpu, dataset_test, transform, threshold):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'out.txt'
    num_images = len(dataset_test)
    # for i in range(num_images):
    for idx in range(NUM):
        index = ids.index(img_ids[idx])

        print(f'testing image {idx + 1} / {num_images} (ids:{index})...')
        img = dataset_test.pull_image(index)
        annos = dataset_test.pull_anno(index)
        height, width, _ = img.shape
        targets = target_transform(annos, width, height)

        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = x.unsqueeze(0)

        if opt.use_gpu and torch.cuda.is_available():
            x = x.cuda()

        y = net(x)      # forward pass
        out = y.data     # shape:[1, 201, 200, 5], 5: x, y, w+x, h+y, class_core

        # scale each detection back up to the image
        scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

        # plot image
        plt.imshow(img)
        # tensor -> pos, scores, classes
        pos, classes, scores = select_data(out, threshold)
        for p, c, s in zip(pos, classes, scores):
            class_name = CLASSES[c]
            score = s
            x1, y1, x2, y2 = p * scale
            
            x, y, w, h = x1, y1, x2-x1, y2-y1
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False))
            txt = f'{class_name}:{round(score*100,2)}%'
            plt.text(x, y, txt)
        for target in targets:     # plot target bbox
            x1, y1, x2, y2 = np.array(target[:4]) * (width, height, width, height)
            x, y, w, h = x1, y1, x2-x1, y2-y1
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=(0., 1., 0.)))
            label_name = CLASSES[target[-1]]
            txt = f'{label_name}'
            plt.text(x, y, txt)
        plt.savefig(f'{opt.out_folder}{time.strftime(f"%Y-%m-%d_%H:%M:%S_{index}.png", time.localtime())}')
        plt.clf()


def test_coco():
    # load net
    num_classes = len(CLASSES) + 1 # +1 background
    net = build_ssd('test', cfg['min_dim'], cfg['num_classes'])
    net.load_state_dict(torch.load(opt.trained_model))
    net.eval()
    print('load model...finished!')
    # load data
    dataset_test = COCODateset(root=opt.root_path, image_set=IMAGE_SET, augment_transform=SSDAugmentation(cfg['min_dim'], MEANS))
    if opt.use_gpu and torch.cuda.is_available():
        print('use gpu!')
        net.cuda()
        torch.backends.cudnn.benchmark = True
    
    # evaluation
    test_net(
        opt.out_folder,
        net,
        opt.use_gpu,
        dataset_test,
        BaseTransform(cfg['min_dim'], MEANS),
        threshold=opt.visual_threshold
    )

if __name__ == '__main__':
    '''
    测试指定类别
    '''
    test_coco()
