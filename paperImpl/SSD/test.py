import os
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
from data.coco import coco_conf as cfg, MEANS
from layers.box_utils import select_data


if opt.use_gpu and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(opt.out_folder):
    os.mkdir(opt.out_folder)

# plot setting
import matplotlib
# all params: ~/venv/torch_venv/lib/python3.6/site-packages/matplotlib/mpl-data/matplotlibrc
matplotlib.rcParams['patch.edgecolor'] = '#0000ff'
matplotlib.rcParams['patch.linewidth'] = 1
# matplotlib.rcParams['font.family'] = 'YouYuan'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['text.color'] = '#ff0000'


def test_net(save_folder, net, use_gpu, dataset_test, transform, threshold):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'out.txt'
    num_images = len(dataset_test)
    for index in range(10):
        print(f'testing image {index + 1} / {num_images}...')
        img = dataset_test.pull_image(index)
        # anno = dataset_test.pull_anno(index)
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
        s_t = time.time()   # timer
        
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
        e_t = time.time()   # timer
        print(f'{e_t - s_t} s') # timer

        plt.savefig(f'{opt.out_folder}{time.strftime(f"%Y-%m-%d_%H:%M:%S_{index}.png", time.localtime())}')
        plt.clf()


def test_coco():
    # load net
    num_classes = len(CLASSES) + 1 # +1 background
    net = build_ssd('test', cfg['min_dim'], cfg['num_classes'])
    if opt.use_gpu and torch.cuda.is_available():
        net.load_state_dict(torch.load(opt.trained_model))
        net.cuda()
        torch.backends.cudnn.benchmark = True
    else:
        net.load_state_dict(torch.load(opt.trained_model, map_location=lambda storage, loc: storage))
    net.eval()
    print('load model...finished!')
    # load data
    dataset_test = COCODateset(root=opt.root_path, augment_transform=SSDAugmentation(cfg['min_dim'], MEANS))

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
    test_coco()
