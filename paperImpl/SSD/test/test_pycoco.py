import os
import sys
os.chdir('..')
sys.path.append('.')

import unittest
from config import opt
import matplotlib.pyplot as plt
import cv2

from pycocotools.coco import COCO

# https://github.com/cocodataset/cocoapi/tree/master/PythonAPI

import matplotlib; matplotlib.use('TkAgg')

class TestPyCOCO(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test_annotations_instances_train2014_json(self):
        '''
            instances_xxx2014.json is used for object detection.
        '''
        print('\n\n===', sys._getframe().f_code.co_name)
        import json
        j = json.load(open(opt.train_annotations_path, 'r'))
        
        print('--- train_annotations keys: \n\t', list(j.keys()))
        print('--- train_annotations nums: \n\t', j['images'].__len__())
        print('--- train_annotations image[0] content: \n\t', j['images'][0])
        print('--- train_annotations annotations[0] content: \n\t', j['annotations'][0])
        print('--- train_annotations annotations[0]["category_id"](class) content: \n\t', j['annotations'][0]['category_id'])
        print('--- train_annotations annotations[0]["bbox"](label) content: \n\t', j['annotations'][0]['bbox'])
        print('--- train_annotations annotations[0] classes name content: \n\t',  j['categories'][0])
        print('--- how link image and annotation(label) by annotation: "./images/COCO_val2014_000000" + j["annotations"][0]["image_id"] : \n\t', os.path.join(opt.train_images_path, 'COCO_val2014_000000' + str(j['annotations'][0]['image_id']) + '.jpg'))


    def test_annotations_captions_train2014_json(self):
        '''
            captions_xxx2014.json is used for index by image or text.
            it is similar with ^up^.
        '''
        print('\n\n===', sys._getframe().f_code.co_name)
        import json
        path = '/' + os.path.join(*opt.train_annotations_path.split('/')[:-1]) + '/captions_train2014.json'
        j = json.load(open(path, 'r'))
        
        print('--- train_annotations captions: \n\t', j['annotations'][0])


    def test_pycoco(self):
        '''
                            ------> loadImgs()
                            |
            getCatIds() -> getImgIds() / catToImgs[]
                |           |
                |           |---> imgToAnns()
                ---------------> getAnnIds() -> loadAnns()

            ps: category_id != CatIds
        '''
        print('\n\n===', sys._getframe().f_code.co_name)

        coco = COCO(opt.train_annotations_path)
        coco_root = opt.train_images_path

        # 利用getCatIds函数获取某个类别对应的ID
        ids = coco.getCatIds('person')[0]
        print(f'--- "person" 对应的序号: {ids}')

        # 获取包含person的所有图片
        imgIds = coco.getImgIds(catIds=[1])
        print(f'--- 包含person的图片共有：{len(imgIds)}张')

        # 利用loadCats获取序号对应的文字类别
        cats = coco.loadCats(1)
        print(f'--- "1" 对应的类别名称: {cats}')

        # 获取包含car的所有图片
        ids = coco.getCatIds(['car'])[0]
        img_ids = coco.catToImgs[ids]
        print(f'--- 包含car的图片共有：{len(imgIds)}')

        # 加载图像
        img_info = coco.loadImgs(img_ids[10])[0]
        img_path = os.path.join(coco_root, img_info['file_name'])
        im = cv2.imread(img_path)
        plt.imshow(im)
        plt.show()

        # 边缘
        plt.imshow(im)
        anno_id = coco.getAnnIds(imgIds=img_info['id']) # 获取该图像对应的anns的Id
        anns = coco.loadAnns(anno_id)
        coco.showAnns(anns)
        plt.show()

        # 蒙版
        mask = coco.annToMask(anns[0])
        plt.imshow(mask)
        plt.show()

        # bbox标注
        '''
          (x, y) ______
                |      |
                |      |
                |______|(x+w, y+h)
        '''
        x, y, w, h = anns[0]['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255))
        plt.imshow(im)
        plt.show()

        import ipdb; ipdb.set_trace()



if __name__ == "__main__":
    unittest.main()
