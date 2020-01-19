import os
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import cv2

# COCO_ROOT = os.path.join(HOME, 'data/coco/')
COCO_ROOT = '/home/data/datasets/COCO2014/'
IMAGE_SET = 'train2014'
IMAGES = 'images'
ANNOTATIONS = 'annotations'
COCO_API = 'PythonAPI'
INSTANCES_SET = 'instances_{}.json'
COCO_CLASSES = ('no',
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
    'fire hydrant', '#', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
    'elephant', 'bear', 'zebra', 'giraffe', '#', 'backpack', 'umbrella', '#', '#', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', '#', 
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', '#', 
    'dining table', '#', '#', 'toilet', '#', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
    'toaster', 'sink', 'refrigerator', '#', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
)
COCO_CLASSES_CN = ('无',
    '人', '自行车', '汽车', '摩托车', '飞机', '公共汽车', '火车', '卡车', '船', '交通信号灯', 
    '消火栓', '停车标志', '停车收费表', '长凳', '鸟', '猫', '狗', '马', '绵羊', '牛', 
    '大象', '熊', '斑马', '长颈鹿', '背包', '雨伞', '手提包', '领带', '手提箱', '飞盘', 
    '滑雪板', '滑雪板', '运动球', '风筝', '棒球棍', '棒球手套', '滑板', '冲浪板', '网球拍', '瓶子', 
    '酒杯', '杯子', '叉子', '菜刀', '勺子', '碗', '香蕉', '苹果', '三明治', '橙色', 
    '西兰花', '胡萝卜', '热狗', '比萨', '甜甜圈', '蛋糕', '椅子', '沙发', '盆栽植物', '床', 
    '餐桌', '厕所', '电子电', '笔记本电脑', '鼠标', '远程的', '键盘', '手机', '微波炉', '烤箱', 
    '烤面包机', '水槽', '冰箱', '书', '时钟', '花瓶', '剪刀', '泰迪熊', '吹风机', '牙'
)

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))
MEANS = (104, 117, 123)
# SSD300 CONFIGS
coco_conf = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}


def detection_collate(batch):
    """
    因为每张图片的target数量不一样，无法使用默认DataLoader的参数collate_fn默认方式简单torch.stack()，
    故手动实现，数据的聚合，img使用stack，target使用list。

    Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = obj['category_id']
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("no bbox problem!")

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class COCODateset(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    # image_set='trainval35k'
    def __init__(self, root, image_set=IMAGE_SET, augment_transform=None,
                 target_transform=COCOAnnotationTransform(), dataset_name='MS COCO'):
        # sys.path.append(os.path.join(root, COCO_API))
        from pycocotools.coco import COCO
        self.root = os.path.join(root, IMAGES, image_set)
        self.coco = COCO(os.path.join(root, ANNOTATIONS, INSTANCES_SET.format(image_set)))
        self.ids = list(self.coco.imgToAnns.keys())
        self.augment_transform = augment_transform
        self.target_transform = target_transform
        self.name = dataset_name

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        img_id = self.ids[index]
        target = self.coco.imgToAnns[img_id]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        target = self.coco.loadAnns(ann_ids)
        path = os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        assert os.path.exists(path), 'Image path does not exist: {}'.format(path)
        img = cv2.imread(os.path.join(self.root, path))
        height, width, _ = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.augment_transform is not None:
            target = np.array(target)
            img, boxes, labels = self.augment_transform(img, target[:, :4], target[:, 4])
            img = img[:, :, (2, 1, 0)]  # to rgb

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR)[:, :, (2, 1, 0)]

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
