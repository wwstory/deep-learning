import warnings

class Config(object):

    # dataset path
    root_path = '/home/data/datasets/COCO2014/'
    train_images_path = root_path + 'images/train2014'
    valid_images_path = root_path + 'images/val2014'
    train_annotations_path = root_path + 'annotations/instances_train2014.json'
    valid_annotations_path = root_path + 'annotations/instances_val2014.json'
    coco_labels_path = root_path + 'coco_labels.txt'

    # dataset, pretrain
    dataset = 'COCO'
    basenet = '/home/data/pretrain_models/vgg16_reducedfc.pth'
    save_folder = 'checkpoints/'
    trained_model = save_folder + 'COCO.pth' # '/home/data/pretrain_models/COCO.pth'
    
    # net
    max_epoch = 100
    batch_size = 32
    num_workers = 4
    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    gamma = 0.1

    # train
    use_gpu = True
    is_visdom = False
    visual_threshold = 0.3
    
    # other
    out_folder = 'out/'
    every_save = 1
    every_print = 1
    every_valid = 1
    debug = '/tmp/debug'


    def parse(self, **kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('User config:')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = Config()
