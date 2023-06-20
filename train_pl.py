import argparse

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch.backends import cudnn
from datasets.dataset import Crowd
from torch.utils.data.dataloader import default_collate

from trainer_pl import LitVGGDistill
import pytorch_lightning as pl
import os


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    return images, points,


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CSRNet')

    parser.add_argument('--save-dir', default='./',
                        help='directory to save models.')
    parser.add_argument('--data-dir', default='./',
                        help='training data directory')
    parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                        help='path to the pretrained model')
    parser.add_argument('--teacher_ckpt', '-tckpt', default='weights/CSRNet_best_mae-v2.ckpt', type=str,
                        help='path to the pretrained model')
    parser.add_argument('--crop_size', default='512', type=int,
                        help='image crop size')

    args = parser.parse_args()

    datasets = {x: Crowd(os.path.join(args.data_dir, x), phase=x, crop_size=args.crop_size) for x in
                ['train', 'validation']}
    dataloaders = {x: DataLoader(datasets[x],
                                 collate_fn=(train_collate
                                             if x == 'train' else default_collate),
                                 batch_size=1,
                                 shuffle=(True if x == 'train' else False),
                                 pin_memory=(True if x == 'train' else False), drop_last=True)
                   for x in ['train', 'validation']}
    train_loader = dataloaders['train']
    val_loader = dataloaders['validation']
    # mock_model = CSRNet()
    # mock_student = CSRNetStudent()
    # mock_model.eval()
    # mock_student.eval()
    # for img, target in train_loader:
    #     pred = mock_model(img)
    #     pred2 = mock_student(img)
    #     print('PRED teacher: ', pred.shape)
    #     print('PRED student: ', pred2.shape)
    #     print('TARGET: ', target.shape)

    model = LitVGGDistill(teacher_ckpt=args.teacher_ckpt, crop_size=args.crop_size)
    save_best_mae = ModelCheckpoint(
        monitor="val/mae",
        dirpath="weights_distill/",
        filename="vgg_student_best_mae",
        mode='min',
        save_last=True)
    trainer = pl.Trainer(gpus=1, callbacks=[save_best_mae], num_sanity_val_steps=0, default_root_dir='log_distill/',
                         max_epochs=500)
    trainer.fit(model, train_loader, val_loader)
