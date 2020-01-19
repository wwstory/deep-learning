import os
import sys
os.chdir('..')
sys.path.append('.')

import unittest
from config import opt

# data
from data import coco, COCODateset, detection_collate
from utils import SSDAugmentation
from data.config import MEANS
# net
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from ssd import build_ssd
from layers.modules import MultiBoxLoss
# other
from tqdm import tqdm

class TestTrain(unittest.TestCase):

    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_data(self):
        '''
            load coco
        '''
        print('\n\n===', sys._getframe().f_code.co_name)
        cfg = coco
        dataset = COCODateset(root=opt.root_path, transform=SSDAugmentation(cfg['min_dim'], MEANS))


    def test_train(self):
        '''
            create net
        '''
        print('\n\n===', sys._getframe().f_code.co_name)
        # data
        cfg = coco
        dataset = COCODateset(root=opt.root_path, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        train_dataloader = DataLoader(
            dataset, 
            opt.batch_size, 
            num_workers=opt.num_workers, 
            shuffle=True, 
            collate_fn=detection_collate, 
            pin_memory=True
        )

        # net
        cfg = coco
        net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])

        if opt.use_gpu and torch.cuda.is_available():
            net.cuda()

        if os.path.exists(opt.trained_model):
            print('Resuming training, loading {}...'.format(opt.trained_model))
            net.load_weights(torch.load(opt.trained_model))
        else:
            vgg_weights = torch.load(opt.basenet)
            print('Loading base network...')
            net.vgg.load_state_dict(vgg_weights)

            print('Initializing weights...')
            # initialize newly added layers' weights with xavier method
            net.extras.apply(weights_init)
            net.loc.apply(weights_init)
            net.conf.apply(weights_init)

        optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
        criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, opt.use_gpu)

        net.train()

        # train
        print('--- start train')
        for epoch in range(opt.max_epoch):
            for _, (i, (images, targets)) in zip(tqdm(range(len(train_dataloader))), enumerate(train_dataloader)):
                if opt.use_gpu and torch.cuda.is_available():
                    images = torch.tensor(images).cuda()
                    targets = [torch.tensor(ann).cuda().requires_grad_(False) for ann in targets]
                else:
                    images = torch.tensor(images)
                    targets = [torch.tensor(ann).requires_grad_(False) for ann in targets]

                out = net(images)
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
                loss.backward()
                optimizer.step()
            
            # print log
            if epoch % opt.every_print == 0:
                print(f'iter {epoch} || Loss: {loss.data}')
            # save
            if epoch % opt.every_save == 0:
                torch.save(net.state_dict(), opt.save_folder + opt.dataset + '.pth')
            # change lr
            if epoch in cfg['lr_steps']:
                adjust_learning_rate(optimizer, opt.gamma, epoch)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def xavier(param):
    import torch.nn.init as init
    init.xavier_uniform(param)


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    unittest.main()
