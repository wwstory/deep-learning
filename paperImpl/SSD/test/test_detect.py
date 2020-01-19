import os
import sys
os.chdir('..')
sys.path.append('.')

# need
import numpy as np
import torch
# net
from ssd import build_ssd
# data
from data import COCO_CLASSES as CLASSES
from data import BaseTransform
# other
from config import opt
from data.coco import coco_conf as cfg, MEANS
from layers.box_utils import select_data
# test
import time
import matplotlib.pyplot as plt


if opt.use_gpu and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class Detect:

    net = None
    transform = None

    THRESHOLD = opt.visual_threshold
    IN_IMAGE_WIDTH = 640
    IN_IMAGE_HEIGHT = 480

    def __init__(self):
        # net
        self.net = build_ssd('test', cfg['min_dim'], cfg['num_classes'])
        if opt.use_gpu and torch.cuda.is_available():
            self.net.load_state_dict(torch.load(opt.trained_model))
            self.net.cuda()
            torch.backends.cudnn.benchmark = True
        else:
            self.net.load_state_dict(torch.load(opt.trained_model, map_location=lambda storage, loc: storage))
        self.net.eval()
        # data
        self.transform = BaseTransform(cfg['min_dim'], MEANS)

    def __call__(self, image):
        return self.detect(image)

    def detect(self, img):
        '''
        in：numpy [h, w, 3]
        out：list [[x1, y1, x2, y2, classes, score], ...]
        '''
        x = torch.from_numpy(self.transform(img)[0]).permute(2, 0, 1).unsqueeze(0)
        
        if opt.use_gpu and torch.cuda.is_available():
            x = x.cuda()
        
        out = self.net(x).data
        pos, classes, scores = select_data(out, self.THRESHOLD)

        detection = []
        for p, c, s in zip(pos, classes, scores):
            x1, y1, x2, y2 = p
            detection.append([x1, y1, x2, y2, c, s])
        
        return detection


if __name__ == '__main__':
    # plot setting
    import matplotlib
    # all params: ~/venv/torch_venv/lib/python3.6/site-packages/matplotlib/mpl-data/matplotlibrc
    matplotlib.rcParams['patch.edgecolor'] = '#0000ff'
    matplotlib.rcParams['patch.linewidth'] = 1
    # matplotlib.rcParams['font.family'] = 'YouYuan'
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['text.color'] = '#ff0000'


    from PIL import Image
    img = np.array(Image.open('/tmp/test.jpg'))
    detector = Detect()
    detection = detector(img)
    print(detection)

    # plot
    plt.imshow(img)
    scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    for d in detection:
        class_name = CLASSES[d[4]]
        score = d[-1]
        x1, y1, x2, y2 = np.array(d[:4]) * scale
            
        x, y, w, h = x1, y1, x2-x1, y2-y1
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False))
        txt = f'{class_name}:{round(score*100,2)}%'
        plt.text(x, y, txt)
    plt.show()
