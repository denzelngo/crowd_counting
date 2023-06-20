import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from geomloss import SamplesLoss
from models.vgg_teacher import VGG
from models.vgg_student import VGG19Student
from torchmetrics import MeanAbsoluteError, MeanSquaredError
import numpy as np


class LitVGGTrainer(pl.LightningModule):
    def __init__(self, crop_size=832):
        super().__init__()

        self.downsample_ratio = 8
        self.blur = 0.01
        self.tau = 0.1
        self.scaling = 0.5
        self.reach = 0.5
        self.p = 1
        self.crop_size = crop_size
        self.d_point = 'l1'
        self.d_pixel = 'l2'

        self.model = VGG()

        self.train_mae = MeanAbsoluteError(compute_on_step=False)
        self.train_mse = MeanSquaredError(compute_on_step=False)

        self.val_mae = MeanAbsoluteError(compute_on_step=False)
        self.val_mse = MeanSquaredError(compute_on_step=False)

        self.criterion = SamplesLoss(blur=self.blur, scaling=self.scaling, debias=False, backend='tensorized',
                                     cost=self.cost, reach=self.reach, p=self.p)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 1e-5, weight_decay=1e-5)
        return optimizer

    def cost(self, X, Y):
        x_col = X.unsqueeze(-2)
        y_lin = Y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** 2, -1)
        C = torch.sqrt(C)
        s = (x_col[:, :, :, -1] + y_lin[:, :, :, -1]) / 2
        s = s * 0.2 + 0.5
        return torch.exp(C / s) - 1

    def training_step(self, batch, batch_idx):
        inputs, points = batch
        if inputs.shape[-1] != self.crop_size:
            self.crop_size = inputs.shape[-1]

        output = self.model(inputs)

        gd_count = torch.tensor([len(p) for p in points], dtype=torch.float32)

        shape = (inputs.shape[0], int(inputs.shape[2] / self.downsample_ratio),
                 int(inputs.shape[3] / self.downsample_ratio))

        cood_grid = grid(output.shape[2], output.shape[3], 1).unsqueeze(0) * self.downsample_ratio + (
                self.downsample_ratio / 2)
        cood_grid = cood_grid / float(self.crop_size)
        i = 0
        emd_loss = 0
        point_loss = 0
        pixel_loss = 0
        entropy = 0
        for p in points:
            if len(p) < 1:
                gt = torch.zeros((1, shape[1], shape[2])).cuda()
                point_loss += torch.abs(gt.sum() - output[i].sum()) / shape[0]
                pixel_loss += torch.abs(gt.sum() - output[i].sum()) / shape[0]
                emd_loss += torch.abs(gt.sum() - output[i].sum()) / shape[0]
            else:
                gt = torch.ones((1, len(p), 1)).cuda()
                cood_points = p.reshape(1, -1, 2) / float(self.crop_size)
                A = output[i].reshape(1, -1, 1)
                l, F, G = self.criterion(A, cood_grid.cuda(), gt, cood_points.cuda())

                C = self.cost(cood_grid.cuda(), cood_points.cuda())
                PI = torch.exp((F.repeat(1, 1, C.shape[2]) + G.permute(0, 2, 1).repeat(1, C.shape[1],
                                                                                       1) - C).detach() / self.blur ** self.p) * A * gt.permute(
                    0, 2, 1)
                entropy += torch.mean((1e-20 + PI) * torch.log(1e-20 + PI))
                emd_loss += (torch.mean(l) / shape[0])
                if self.d_point == 'l1':
                    point_loss += torch.sum(torch.abs(PI.sum(1).reshape(1, -1, 1) - gt)) / shape[0]
                else:
                    point_loss += torch.sum((PI.sum(1).reshape(1, -1, 1) - gt) ** 2) / shape[0]
                if self.d_pixel == 'l1':
                    pixel_loss += torch.sum(torch.abs(PI.sum(2).reshape(1, -1, 1).detach() - A)) / shape[0]
                else:
                    pixel_loss += torch.sum((PI.sum(2).reshape(1, -1, 1).detach() - A) ** 2) / shape[0]
            i += 1
        loss = emd_loss + self.tau * (pixel_loss + point_loss) + self.blur * entropy
        self.log('train/loss_total', loss, on_step=False, on_epoch=True)

        self.log('train/loss_emd', emd_loss, on_step=False, on_epoch=True)
        self.log('train/loss_pixel', pixel_loss, on_step=False, on_epoch=True)
        self.log('train/loss_point', point_loss, on_step=False, on_epoch=True)

        output = torch.mean(output, dim=1)
        pre_count = torch.sum(output[-1]).detach()

        self.train_mae.update(pre_count, gd_count[-1])
        self.train_mse.update(pre_count, gd_count[-1])

        self.log('train_metric/mae', self.train_mae.compute(), on_step=False, on_epoch=True)
        self.log('train_metric/mse', torch.sqrt(self.train_mse.compute()), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, val_batch, val_batch_idx):
        inputs, points = val_batch
        self.model.eval()
        pred = self.model(inputs)

        gd_count = torch.tensor([len(p) for p in points], dtype=torch.float32)

        self.val_mae.update(pred.data.sum(), gd_count[-1])
        self.val_mse.update(pred.data.sum(), gd_count[-1])
        self.log('val/mae', self.val_mae.compute(), on_step=False, on_epoch=True)
        self.log('val/mse', torch.sqrt(self.val_mse.compute()), on_step=False, on_epoch=True)


def grid(H, W, stride):
    coodx = torch.arange(0, W, step=stride) + stride / 2
    coody = torch.arange(0, H, step=stride) + stride / 2
    y, x = torch.meshgrid([coody / 1, coodx / 1])
    return torch.stack((x, y), dim=2).view(-1, 2)
