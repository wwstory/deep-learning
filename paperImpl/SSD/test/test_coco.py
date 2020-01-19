import os
import sys
os.chdir('..')
sys.path.append('.')

import unittest
from config import opt
from tqdm import tqdm

from data import COCODateset, detection_collate
# from data import SSDAugmentation
from _augmentations import SSDAugmentation  # 修改移除了Resize的数据增强方式
from data.coco import coco_conf as cfg, MEANS, COCO_CLASSES

import numpy as np
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import matplotlib.pyplot as plt
import cv2
import matplotlib; matplotlib.use('TkAgg')


class TestCOCO(unittest.TestCase):

    def setUp(self):
        pass


    def tearDown(self):
        pass


    def xtest_coco_dataloader(self):
        '''
            coco loader
        '''
        print('\n\n===', sys._getframe().f_code.co_name)
        dataset = COCODateset(root=opt.root_path, augment_transform=SSDAugmentation(cfg['min_dim'], MEANS))
        train_dataloader = DataLoader(
            dataset, 
            opt.batch_size, 
            num_workers=opt.num_workers, 
            shuffle=True, 
            drop_last=True,
            collate_fn=detection_collate, 
            # pin_memory=True
        )

        for epoch in range(opt.max_epoch):
            # for _, (i, (images, targets)) in zip(tqdm(range(len(train_dataloader))), enumerate(train_dataloader)):
            for _, (i, (images, targets)) in zip(tqdm(range(3)), enumerate(train_dataloader)):
                print(images.shape)
                len(targets)
                # if opt.use_gpu:
                #     images = torch.tensor(images).cuda()
                #     targets = [torch.tensor(ann).cuda().requires_grad_(False) for ann in targets]
                # else:
                #     images = torch.tensor(images)
                #     targets = [torch.tensor(ann).requires_grad_(False) for ann in targets]
                

    def xtest_coco_dataset(self):
        '''
            coco dataset
        '''
        print('\n\n===', sys._getframe().f_code.co_name)

        dataset = COCODateset(root=opt.root_path, augment_transform=SSDAugmentation(cfg['min_dim'], MEANS))

        import ipdb; ipdb.set_trace()


    def test_plot_coco(self):
        '''
            plot coco
        '''
        print('\n\n===', sys._getframe().f_code.co_name)

        dataset = COCODateset(root=opt.root_path, augment_transform=SSDAugmentation(cfg['min_dim'], MEANS))
        train_dataloader = DataLoader(
            dataset, 
            opt.batch_size, 
            num_workers=opt.num_workers, 
            shuffle=True, 
            drop_last=True,
            collate_fn=detection_collate, 
            # pin_memory=True
        )

        # for epoch in range(opt.max_epoch):
        #     for _, (i, (images, targets)) in zip(tqdm(range(len(train_dataloader))), enumerate(train_dataloader)):
        for epoch in range(1):
            for _, (i, (images, targets)) in zip(tqdm(range(1)), enumerate(train_dataloader)):
                # import ipdb; ipdb.set_trace()
                import matplotlib; matplotlib.use('TkAgg')
                img = images[0]
                target = targets[0]
                transform = T.ToPILImage()
                img = transform(img)
                img = np.array(img)
                target = resume_scale(target, img)  # resume
                boxes = target[:, : 4]
                plt.imshow(img)
                for box, ta in zip(boxes, target):
                    x1, y1, x2, y2 = box
                    print('-box位置：', box, end='')
                    print('-> class类别:', COCO_CLASSES[int(ta[-1])])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
                    txt = COCO_CLASSES[int(ta[-1])]
                    cv2.putText(img, txt, org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 70, 70))
                plt.imshow(img)
                plt.show()
                

def resume_scale(target, image):
    width = image.shape[1]
    height = image.shape[0]
    scale = torch.Tensor([width, height, width, height, 1])
    target = target * scale
    return target


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    unittest.main()
