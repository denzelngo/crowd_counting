import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from geomloss import SamplesLoss
from models.vgg_teacher import VGG
from models.vgg_student import VGG19Student
from pytorch_lightning.metrics import MeanAbsoluteError, MeanSquaredError
import numpy as np


class LitVGGDistill(pl.LightningModule):
    def __init__(self, teacher_ckpt=None, crop_size=512):
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

        self.teacher = VGG()
        self.student = VGG19Student()
        self.teacher.regist_hook()  # use hook to get teacher's features

        if teacher_ckpt:
            self.teacher.load_state_dict(torch.load(teacher_ckpt))
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.train_mae = MeanAbsoluteError(compute_on_step=False)
        self.train_mse = MeanSquaredError(compute_on_step=False)

        self.val_mae = MeanAbsoluteError(compute_on_step=False)
        self.val_mse = MeanSquaredError(compute_on_step=False)

        self.criterion = SamplesLoss(blur=self.blur, scaling=self.scaling, debias=False, backend='tensorized',
                                     cost=self.cost, reach=self.reach, p=self.p)
        self.mse = torch.nn.MSELoss(size_average=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student.parameters(), 1e-5, weight_decay=1e-5)
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
        # when the min size of image less than crop size
        if inputs.shape[-1] != self.crop_size:
            self.crop_size = inputs.shape[-1]
        gd_count = torch.tensor([len(p) for p in points], dtype=torch.float32)

        with torch.no_grad():
            teacher_output = self.teacher(inputs)
            # self.teacher.features_list.append(teacher_output)
            # teacher_fsp_features = [scale_process(self.teacher.features_list)]
            # teacher_fsp = cal_dense_fsp(teacher_fsp_features)

        student_features = self.student(inputs)
        outputs = student_features[-1]
        # student_fsp_features = [scale_process(student_features)]
        # student_fsp = cal_dense_fsp(student_fsp_features)

        loss_s = self.mse(outputs, teacher_output)

        # loss_f = []
        # assert len(teacher_fsp) == len(student_fsp)
        # for t in range(len(teacher_fsp)):
        #     loss_f.append(self.mse(teacher_fsp[t], student_fsp[t]))
        # loss_fsp = sum(loss_f) / len(teacher_fsp) * 0.1
        #
        # loss_c = []
        # for t in range(len(student_features) - 1):
        #     loss_c.append(cosine_similarity(student_features[t], self.teacher.features_list[t]))
        # loss_cos = sum(loss_c) * 0.1

        shape = (inputs.shape[0], int(inputs.shape[2] / self.downsample_ratio),
                 int(inputs.shape[3] / self.downsample_ratio))

        cood_grid = grid(outputs.shape[2], outputs.shape[3], 1).unsqueeze(0) * self.downsample_ratio + (
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
                point_loss += torch.abs(gt.sum() - outputs[i].sum()) / shape[0]
                pixel_loss += torch.abs(gt.sum() - outputs[i].sum()) / shape[0]
                emd_loss += torch.abs(gt.sum() - outputs[i].sum()) / shape[0]
            else:
                gt = torch.ones((1, len(p), 1)).cuda()
                cood_points = p.reshape(1, -1, 2) / float(self.crop_size)
                A = outputs[i].reshape(1, -1, 1)
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
        loss = emd_loss + self.tau * (pixel_loss + point_loss) + self.blur * entropy + loss_s
        self.log('train/loss_total', loss, on_step=False, on_epoch=True)
        # self.log('train/loss_fsp', loss_fsp, on_step=False, on_epoch=True)
        # self.log('train/loss_cosine', loss_cos, on_step=False, on_epoch=True)
        self.log('train/loss_soft', loss_s, on_step=False, on_epoch=True)
        self.log('train/loss_emd', emd_loss, on_step=False, on_epoch=True)
        self.log('train/loss_pixel', pixel_loss, on_step=False, on_epoch=True)
        self.log('train/loss_point', point_loss, on_step=False, on_epoch=True)

        outputs = torch.mean(outputs, dim=1)
        pre_count = torch.sum(outputs[-1]).detach()

        self.train_mae.update(pre_count, gd_count[-1])
        self.train_mse.update(pre_count, gd_count[-1])

        self.log('train_metric/mae', self.train_mae.compute(), on_step=False, on_epoch=True)
        self.log('train_metric/mse', torch.sqrt(self.train_mse.compute()), on_step=False, on_epoch=True)

        self.log('train_metric/mae', self.train_mae.compute(), on_step=False, on_epoch=True)
        self.log('train_metric/mse', torch.sqrt(self.train_mse.compute()), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, val_batch, val_batch_idx):
        inputs, points = val_batch
        self.student.eval()
        pred = self.student(inputs)

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


def scale_process(features, ceil_mode=True):
    # process features for multi-scale dense fsp
    scale = [4, 3, 2, 1, 0, 1, 1]
    new_features = []
    for i in range(len(features)):
        if scale[i] == 0:
            new_features.append(features[i])
            continue
        down_ratio = pow(2, scale[i])
        pool = torch.nn.MaxPool2d(kernel_size=down_ratio, stride=down_ratio, ceil_mode=ceil_mode)
        new_features.append(pool(features[i]))
    return new_features


def cosine_similarity(stu_map, tea_map):
    c = stu_map.shape[1]
    similiar = 1 - F.cosine_similarity(stu_map, tea_map, dim=1)
    loss = similiar.sum() / c
    return loss


def cal_dense_fsp(features):
    fsp = []
    for groups in features:
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                x = groups[i]
                y = groups[j]

                norm1 = torch.nn.InstanceNorm2d(x.shape[1])
                norm2 = torch.nn.InstanceNorm2d(y.shape[1])
                x = norm1(x)
                y = norm2(y)
                res = gram(x, y)
                fsp.append(res)
    return fsp


def gram(x, y):
    n = x.shape[0]
    c1 = x.shape[1]
    c2 = y.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    x = x.view(n, c1, -1, 1)[0, :, :, 0]
    y = y.view(n, c2, -1, 1)[0, :, :, 0]
    y = y.transpose(0, 1)
    # print x.shape
    # print y.shape
    z = torch.mm(x, y) / (w * h)
    return z
