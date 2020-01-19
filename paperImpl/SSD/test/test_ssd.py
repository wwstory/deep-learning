import os
import sys
os.chdir('..')
sys.path.append('.')

import unittest
from config import opt

from ssd import *

class TestSSD(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ssd(self):
        '''
        '''
        print('\n\n===', sys._getframe().f_code.co_name)

        phase = 'test'
        size = 300
        num_classes = 201

        if phase != "test" and phase != "train":
            print("ERROR: Phase: " + phase + " not recognized")
            return
        if size != 300:
            print("ERROR: You specified size " + repr(size) + ". However, " +
                "currently only SSD300 (size=300) is supported!")
            return

        param1 = vgg(base[str(size)], 3)
        param2 = add_extras(extras[str(size)], 1024)
        param3 = mbox[str(size)]

        for p in [param1, param2, param3]:
            print('-----------')
            for l in p:
                print(l)

        base_, extras_, head_ = multibox(
            param1,
            param2,
            param3,
            num_classes
        )
        import ipdb; ipdb.set_trace()
        ssd = SSD(phase, size, base_, extras_, head_, num_classes)



if __name__ == "__main__":
    unittest.main()
