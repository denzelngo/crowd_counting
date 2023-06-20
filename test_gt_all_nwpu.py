import scipy.io as io
import os
import torch
from torchvision import transforms
import numpy as np
import cv2

which = 'teacher'

if which == 'student':
    from trainer_pl import LitVGGDistill
    from models.vgg_student import VGG19Student

    lit_vgg_distill = LitVGGDistill.load_from_checkpoint('vgg_student_best_mae_12.ckpt')

    model = VGG19Student()
    model.load_state_dict(lit_vgg_distill.student.state_dict())

    # checkpoint = torch.load('partB_student.pth.tar')
    # model.load_state_dict(checkpoint['state_dict'])
else:
    from models.vgg_teacher import VGG
    from trainer_teacher_pl import LitVGGTrainer

    lit_teacher = LitVGGTrainer.load_from_checkpoint('vgg_teacher_best_mae.ckpt')
    # lit_csrnet = LitCSRNet.load_from_checkpoint('CSRNet_best_mae-v2.ckpt')
    #
    # model = lit_csrnet.net

    model = VGG()
    model.load_state_dict(lit_teacher.model.state_dict())

#
model = model.cuda()
model.eval()

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

# out_pth = 'SH_part_A_train.txt'
img_path = ''
gt_path = ''
img_listdir = os.listdir(img_path)
n = len(img_listdir)
img_listdir.sort(key=lambda x: int(x[:-4]))

img_to_save = []
n_mea = 0
s_mea = 0
with torch.no_grad():
    for img_name in img_listdir:

        gt_file_name = img_name[:-4] + '.mat'
        gt_file = os.path.join(gt_path, gt_file_name)
        img_file = os.path.join(img_path, img_name)
        mat = io.loadmat(gt_file)
        gt_count = len(mat["annPoints"].astype(np.float32))

        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img).cuda()
        output = model(img.unsqueeze(0))
        pred = int(output.data.sum().item())

        mae = abs(pred - gt_count)
        s_mea += mae
        if mae / gt_count > 0.1:
            print(img_name, end='  ')
            print(mae / gt_count, end='  ')
            print(gt_count, pred)
        n_mea += mae / gt_count

        if mae < 20:
            img_to_save.append(img_name)
print('Normalized MAE: ', n_mea / n)
print('MAE: ', s_mea / n)
print(img_to_save)
