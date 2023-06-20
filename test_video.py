from torchvision import transforms
from matplotlib import pyplot as plt
from class_camera import camera
from matplotlib import cm as c
import numpy as np
import torch
import cv2
import time
from models.vgg import vgg19

img_file = ""
rd_department = 'rtsp://user:remote456%2B@192.168.1.217/axis-media/media.amp?videocodec=h264'
meeting_room = 'rtsp://user:remote456%2B@192.168.1.215/axis-media/media.amp?videocodec=h264'
fabrication = 'rtsp://user:remote456%2B@192.168.1.214/axis-media/media.amp?videocodec=h264'
living_room = 'rtsp://user:remote456%2B@192.168.1.216/axis-media/media.amp?videocodec=h264'

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
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

    # checkpoint = torch.load('best_teacher.pth')
    # checkpoint = torch.load('partB_teacher.pth.tar')
    # model.load_state_dict(checkpoint)

model = model.cuda()
model.eval()
# load camera
# cam = camera("crowd_counting_demo2.mp4")
cam = camera('/home/user5/Downloads/VIDÉO_0172.mov')
# cam = camera(fabrication)
print(f"Camera is alive?: {cam.p.is_alive()}")
# out = cv2.VideoWriter('Out_IMG_6745.avi', cv2.VideoWrite<r_fourcc('M', 'J', 'P', 'G'), 10,(frame_width, frame_height))

WINDOW_NAME = "Crowd Count"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 700, 700)

resize = 0.5

# cv2.namedWindow("Color Map", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Color Map", 700, 700)
with torch.no_grad():
    while 1:  # Frames processing
        # Get Frame
        img_ori = cam.get_frame(1)
        h, w, _ = img_ori.shape
        print('Img size: ', w, h)
        img = cv2.resize(img_ori, None, fx=resize, fy=resize)  # resize img to avoid CUDA OOM
        # img = cv2.resize(img_ori, (1280, 720))  # resize img to avoid CUDA OOM
        img = transform(img).cuda()
        tic = time.time()
        output = model(img.unsqueeze(0))

        print("Predicted Count : ", int(output.data.sum().item()))

        fps = 1 / (time.time() - tic)
        print('FPS: ', fps)
        temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2], output.detach().cpu().shape[3]))
        cmap = (temp - temp.min()) / (temp.max() - temp.min())  # min-max scale
        # plt.imshow(temp, cmap=c.jet)
        # plt.pause(0.1)
        # plt.draw()
        cmap = cv2.resize(temp, (w, h))
        cmap = cv2.applyColorMap((cmap * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
        img_show = cv2.add(img_ori, cmap)

        # cv2.imshow("Color Map", cmap)

        cv2.imshow("Crowd Count", img_show)
        key = cv2.waitKey(1)
        if key == 27:  # 13 is the Enter Key / 27 is the Esc key
            break
