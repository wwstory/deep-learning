import os
import sys
os.chdir('..')
sys.path.append('.')

import unittest
from config import opt

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from layers.box_utils import *
from data import coco_conf as cfg
from layers import PriorBox


class TestBoxUtils(unittest.TestCase):

    def setUp(self):
        self.img = np.zeros((300, 300, 3))
        priorbox = PriorBox(cfg)
        self.priors = torch.Tensor(priorbox.forward()).requires_grad_(False)

        self.truths = torch.tensor([
            [0.8041, 0.3796, 1.0000, 0.6487],
            [0.2017, 0.2984, 0.3219, 0.6450],
            [0.4068, 0.0736, 0.5075, 0.2580],
            [0.9694, 0.1445, 1.0000, 0.3685]
        ])

    def tearDown(self):
        print('\n---finish!')

    def test_point_form(self):
        '''
        '''
        print('\n\n===', sys._getframe().f_code.co_name)
        img = self.img
        priors = self.priors
        p_priors = point_form(priors)
        DELAY = 0.1
        color = (1., 1., 1.)
        d_boxes = []
        p_boxes = []

        for i in range(10):
            d_boxes.append(priors[i*100])
            p_boxes.append(p_priors[i*100])

        # print
        plt.imshow(img)
        for d, p in zip(d_boxes, p_boxes):
            x, y, w, h = d * 300
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=1, edgecolor=color))
            plt.pause(DELAY)
            x1, y1, x2, y2 = p * 300
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, linewidth=1, edgecolor=color))
            plt.pause(DELAY*3)

            print(f'({x1}, {y1}) ({x2}, {y2})')
            color = (random.random(), random.random(), random.random())
        plt.show()


    def test_intersect(self):
        '''
        '''
        print('\n\n===', sys._getframe().f_code.co_name)
        box_a = self.truths
        box_b = self.priors
        intersect(box_a, box_b)


    def test_jaccard(self):
        '''
        '''
        print('\n\n===', sys._getframe().f_code.co_name)
        truths = self.truths
        priors = self.priors

        overlaps = jaccard(
            truths,
            point_form(priors)
        )


    def test_match(self):
        '''
        '''
        print('\n\n===', sys._getframe().f_code.co_name)


    def test_encode(self):
        '''
        '''
        print('\n\n===', sys._getframe().f_code.co_name)


    def test_decode(self):
        '''
        '''
        print('\n\n===', sys._getframe().f_code.co_name)


    def test_log_sum_exp(self):
        '''
        '''
        print('\n\n===', sys._getframe().f_code.co_name)


    def test_nms(self):
        '''
        '''
        print('\n\n===', sys._getframe().f_code.co_name)



if __name__ == "__main__":
    unittest.main()
