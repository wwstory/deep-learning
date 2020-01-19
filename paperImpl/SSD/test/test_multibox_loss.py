import os
import sys
os.chdir('..')
sys.path.append('.')

import unittest
from config import opt
import random

import torch

from data import coco_conf as cfg
from layers.modules import MultiBoxLoss


class TestMultiboxLoss(unittest.TestCase):

    def setUp(self):
        self.targets = [torch.rand(random.randint(1, 10), 5) for i in range(32)]    # 随机1~10个目标，32个样本
        self.out = (
            torch.rand(32, 8732, 4),    # 预测的位置
            torch.rand(32, 8732, 201),  # 预测的类别
            torch.rand(8732, 4),        # 默认(先行)盒位置
        )

    def tearDown(self):
        pass

    def test_multibox_loss(self):
        '''
        '''
        print('\n\n===', sys._getframe().f_code.co_name)

        targets = self.targets
        out = self.out
        criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, opt.use_gpu)
        loss_l, loss_c = criterion(out, targets)



if __name__ == "__main__":
    unittest.main()
