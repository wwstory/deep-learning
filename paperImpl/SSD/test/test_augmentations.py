import os
import sys
os.chdir('..')
sys.path.append('.')

import unittest
from config import opt

import numpy as np
import cv2
import torch
from data import SSDAugmentation
from data.coco import COCOAnnotationTransform
import matplotlib.pyplot as plt
from data.coco import COCO_CLASSES


class TestAugmentations(unittest.TestCase):

    def setUp(self):
        self.img_cv = np.random.randint(0, 255, (720, 1080, 3))
        self.img = self.img_cv
        self.target = [
            {
                "segmentation": [
                    [616.25, 280.63, 606.25, 303.13, 590, 351.88, 578.75, 408.13, 568.75, 474.38, 587.5, 469.38, 633.75, 298.13, 632.5, 283.13]
                ], 
                "area": 4724.21875, 
                "iscrowd": 0, 
                "image_id": 181064, 
                "bbox": [568.75, 280.63, 65, 193.75], 
                "category_id": 31, 
                "id": 1833965
            }, 
            {
                "segmentation": [
                    [639.29, 283.04, 612.78, 281.3, 597.16, 275.6, 596.92, 265.44, 602.86, 257.76, 612.53, 248.34, 619.96, 243.88, 625.41, 243.64, 640, 255.53, 640, 277.58]
                ], 
                "area": 1294.2286500000012, 
                "iscrowd": 0, 
                "image_id": 181064, 
                "bbox": [596.92, 243.64, 43.08, 39.4], 
                "category_id": 1, 
                "id": 2003753
            }
        ]
        # self.target = np.array([[0.1, 0.1, 0.5, 0.5, 3], [0.2, 0.2, 0.7, 0.7, 10]])

    def tearDown(self):
        pass

    def test_augmentations(self):
        '''
            输入numpy类型
        '''
        print('\n\n===', sys._getframe().f_code.co_name)

        img = self.img
        target = self.target
        target_transform = COCOAnnotationTransform()
        height, width, _ = img.shape
        target = target_transform(target, width, height)
        augment_transform = SSDAugmentation()
        target = np.array(target)
        img, boxes, labels = augment_transform(img, target[:, :4], target[:, 4])
        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        target = torch.FloatTensor(target)
        
        target = resume_scale(target, img)  # resume
        boxes = target[:, : 4]
        plt.imshow(img)
        for box, ta in zip(boxes, target):
            x1, y1, x2, y2 = box
            print('-box位置：', box, end='')
            print('-> class类别:', COCO_CLASSES[int(ta[-1])])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
            txt = COCO_CLASSES[int(ta[-1])]
            cv2.putText(img, txt, org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 70, 70), thickness=1)
        plt.imshow(img)
        plt.show()


def resume_scale(target, image):
    width = image.shape[1]
    height = image.shape[0]
    scale = torch.Tensor([width, height, width, height, 1])
    target = target * scale
    return target



if __name__ == "__main__":
    unittest.main()
