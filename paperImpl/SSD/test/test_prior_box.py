import os
import sys
os.chdir('..')
sys.path.append('.')
import random

import unittest
from config import opt

import torch
from data import coco_conf as cfg
from layers import Detect, PriorBox, L2Norm

import matplotlib.pyplot as plt
import cv2
import numpy as np


class TestPriorBox(unittest.TestCase):

    def setUp(self):
        self.img = np.zeros((300, 300, 3))
        self.scale = (300, 300) # (h, w)
        self.shape = (300, 300) # (h, w)

    def tearDown(self):
        pass

    def test_prior_box(self):
        '''
            动画画出feature的default(prior) box的样子。

            Input:
                输入 0|1|2|3|4|5, 画出这6层之一的box。
        '''
        print('\n\n===', sys._getframe().f_code.co_name)

        img = self.img
        DELAY = 0.1
        LAYER = int(input('input -> x=[0|1|2|3|4|5]:') or 0)   # 0, 1, 2, 3, 4, 5
        layer = [38*38*4, 19*19*6, 10*10*6, 5*5*6, 3*3*4, 1*1*4]
        index_layer = [0, 5776, 7942, 8542, 8692, 8728, 8732]
        color = (1., 1., 1.)

        priorbox = PriorBox(cfg)
        priors = torch.Tensor(priorbox.forward()).requires_grad_(False)

        plt.imshow(img)
        for box in priors[index_layer[LAYER]:index_layer[LAYER+1]]:
            cx, cy, w, h = box * 300
            x, y = cx-1/2*w, cy-1/2*h
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=1, edgecolor=color))
            plt.pause(DELAY)

            print(f'({x}, {y}) ({w}, {h})')
            color = (random.random(), random.random(), random.random())
        print('\n---finish!')
        plt.show()


if __name__ == "__main__":
    unittest.main()
