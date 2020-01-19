import os
import sys
os.chdir('..')
sys.path.append('.')

import unittest
from config import opt

class TestDetection(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_detection(self):
        '''
        '''
        print('\n\n===', sys._getframe().f_code.co_name)



if __name__ == "__main__":
    unittest.main()
