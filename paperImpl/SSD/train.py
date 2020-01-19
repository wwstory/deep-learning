import os
import sys
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
# data
from data import coco_conf as cfg, COCODateset, detection_collate
from data.augmentations import SSDAugmentation
from data.coco import MEANS
# net
from ssd import build_ssd
from layers.modules import MultiBoxLoss
# other
from config import opt
from tqdm import tqdm


def train(**kwargs):
    # data
    train_dataset = COCODateset(root=opt.root_path, augment_transform=SSDAugmentation(cfg['min_dim'], MEANS))
    train_dataloader = DataLoader(
        train_dataset, 
        opt.batch_size, 
        num_workers=opt.num_workers, 
        shuffle=True, 
        collate_fn=detection_collate, 
        # pin_memory=True
    )

    # net
    net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])

    if opt.use_gpu and torch.cuda.is_available():
        net.cuda()
        torch.backends.cudnn.benchmark = True

    if os.path.exists(opt.trained_model):
        print('Load weight {}...'.format(opt.trained_model))
        net.load_state_dict(torch.load(opt.trained_model, map_location=lambda storage, loc: storage))
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
    for epoch in tqdm(range(opt.max_epoch)):
        for _, (i, (images, targets)) in zip(tqdm(range(len(train_dataloader))), enumerate(train_dataloader)):
            if opt.use_gpu and torch.cuda.is_available():
                images = torch.Tensor(images).cuda()
                targets = [torch.Tensor(ann).cuda() for ann in targets]
            else:
                images = torch.Tensor(images)
                targets = [torch.Tensor(ann) for ann in targets]

            out = net(images)
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
        
        # debug
        if os.path.exists(opt.debug):
            import ipdb; ipdb.set_trace()
        # print log
        if epoch % opt.every_print == 0:
            print(f'\nepoch {epoch} || Loss: {loss.data}')
        # save
        if epoch % opt.every_save == 0:
            torch.save(net.state_dict(), opt.save_folder + opt.dataset + '.pth')
        # change lr
        if epoch in cfg['lr_steps']:
            adjust_learning_rate(optimizer, opt.gamma, epoch)
        # valid
        # if epoch % opt.every_valid == 0:
        #     valid(net, train_dataset)


# def valid(net, dataset):
#     import matplotlib.pyplot as plt
#     import cv2
#     import numpy as np
#     from data.coco import coco_conf as cfg, MEANS, COCO_CLASSES as CLASSES
#     from data import BaseTransform

#     log = {}
#     transform = BaseTransform(cfg['min_dim'], MEANS)

#     net.eval()
#     index = np.random.choice(len(dataset))
#     img = img_origin = dataset.pull_image(index)
#     anno = dataset.pull_anno(index)
#     img_id = anno[0]['image_id']
#     img = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
#     img = img.unsqueeze(0)

#     if opt.use_gpu:
#         img = img.cuda()

#     out = net(img)
#     import ipdb; ipdb.set_trace()
    
#     scale = torch.Tensor(
#         [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
#     )
#     for i in range(out.size(1)):
#         for j in range(out.size(2)):
#             if out[0, i, j, 0] >= 0.6:
#                 score = out[0, i, j, 0]
#                 label_name = CLASSES[i]
#                 pos = (out[0, i, j, 1:]*scale).cpu().numpy()
#                 x1, y1, x2, y2 = (pos[0], pos[1], pos[2], pos[3])

#                 # plot image
#                 plt.imshow(img_origin)
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
#                 txt = COCO_CLASSES[int(ta[-1])] + ':' + str(score)
#                 cv2.putText(img, txt, org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 70, 70))
#                 plt.savefig(f'{opt.out_folder}{time.strftime("%Y-%m-%d_%H:%M:%S.png", time.localtime())}')

#     net.train()



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
    lr = opt.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    train()
