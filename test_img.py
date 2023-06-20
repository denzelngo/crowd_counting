import h5py
from torchvision import transforms
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as c
import os
import time

which = 'teacher'

if which == 'student':
    from trainer_pl import LitVGGDistill
    from models.vgg_student import VGG19Student

    lit_vgg_distill = LitVGGDistill.load_from_checkpoint('vgg_student_best_mae_nwpu_last.ckpt')

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
# transform = transforms.Compose(

#     [transforms.ToTensor()])
resize = 1
img_name = 'test_img_client1'
img_ori = cv2.imread(f'{img_name}.jpg')
h_ori, w_ori, _ = img_ori.shape
if os.path.isfile(f'{img_name}.h5'):
    gt_file = h5py.File(f'{img_name}.h5')
    target = np.asarray(gt_file['density'])
    h, w = target.shape
    target = cv2.resize(target, (int(w / 8), int(h / 8)), interpolation=cv2.INTER_CUBIC)
    target = target * 64

# img = cv2.resize(img, (1280, 720))
img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

img = cv2.resize(img, None, fx=resize, fy=resize)  # resize img to avoid CUDA OOM

print('Img size: ', img.shape)
img = transform(img).cuda()
with torch.no_grad():
    for i in range(5):
        tic = time.time()
        output = model(img.unsqueeze(0))
        print("Predicted Count : ", int(output.data.sum().item()))
        fps = 1 / (time.time() - tic)
        print('FPS: ', fps)
# print("Predicted Count : ", int(output.detach().cpu().sum().numpy()))

if os.path.isfile(f'{img_name}.h5'):
    print('GT count: ', int(target.sum()))
# print(output.shape)

temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2], output.detach().cpu().shape[3]))
cmap = (temp - temp.min()) / (temp.max() - temp.min())  # min-max scale
# cmap_tosave = cv2.applyColorMap((temp * 255.0 / temp.max()).astype(np.uint8), cv2.COLORMAP_JET)
# cv2.imwrite('cmap_only_{}_{}.png'.format(which, img_name), cmap_tosave)
cmap = cv2.resize(temp, (w_ori, h_ori), cv2.INTER_NEAREST)
cmap = cv2.applyColorMap((cmap * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
img_save = cv2.add(img_ori, cmap)

fig = plt.imshow(temp, cmap=c.jet)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.pause(0.1)
plt.draw()
# plt.axes('off')
cv2.imwrite('out_map_{}_{}.png'.format(which, img_name), img_save)
plt.savefig('cmap_only_{}_{}.png'.format(which, img_name), dpi=200, pad_inches=0)
if os.path.isfile(f'{img_name}.h5'):
    fig = plt.imshow(target, cmap=c.jet)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.savefig(f'ground_truth_{img_name}.png', dpi=200, pad_inches=0)
