import scipy.io as io
import os
import torch
from torchvision import transforms
import cv2
import math

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

model = model.cuda()
model.eval()

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

img_path = '/media/user5/7C3637AF363768F2/Users/user5/Desktop/NWPU_dataset/test'

img_listdir = os.listdir(img_path)
img_listdir.sort(key=lambda x: int(x[:-4]))
f = open('result_test_nwpu.txt', 'a')
with torch.no_grad():
    for img_name in img_listdir:
        img_name_base = img_name[:-4]
        print('Img ', img_name_base)
        img_file = os.path.join(img_path, img_name)
        img = cv2.imread(img_file)
        h, w, _ = img.shape
        if h*w > 5e6:
            r = math.sqrt(h*w/5e6)
            img = img.resize((int(w / r), int(h / r)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img).cuda()
        output = model(img.unsqueeze(0))
        pred = float(output.data.sum().item())

        text = f'{img_name_base}  {pred}'
        f.write(text + '\n')

f.close()
