import os
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import torch
# from eye_dataset import Dataset, get_preprocessing
import albumentations as albu
import imgviz
import json
from PIL import Image


class_names = ['_background_', 'palpebra superior', 'palpebra inferior',
                           'carunculae lacrimalis', 'conjunctiva', 'cornea']


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)


ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax2d'  # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cpu'

model = torch.load('segmentation/best_model.pth', map_location='cpu')

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
preprocessing = get_preprocessing(preprocessing_fn)

# patients = os.listdir('eye_photo')
#
# if not os.path.exists('eye_photo_test'):
#     os.mkdir('eye_photo_test')
#
# for filename in patients:
#     # print(case)
#     #
#     # for filename in os.listdir(os.path.join('/home/eye/Desktop/segmentation/eye_photo/', case)):
#     #     print(filename, filename.endswith('.jpg'))
#         if filename.endswith('.jpg'):
def seg(img, imgname):
    image_0 = cv2.imread(img)
    h0, w0 = image_0.shape[0:2]

    # 旋转图片
    if h0 > w0:
        image_0 = np.rot90(image_0)
        # image_0 = mmcv.imrotate(image_0, -90, auto_bound=True)
        h0, w0 = image_0.shape[0:2]

    image = cv2.resize(image_0, (512, 512))
    img = preprocessing(image=image)['image']

    x_tensor = torch.from_numpy(img).to(DEVICE).unsqueeze(0)
    pr_mask = model.to(DEVICE).predict(x_tensor)
    pr_label = torch.argmax(pr_mask, dim=1)
    pr_label = pr_label.squeeze().cpu().numpy()

    out_viz_file = os.path.join('/home/ubuntu/wx_test/media/img_segroi/'+os.path.basename(imgname))
    viz = imgviz.label2rgb(
        label=pr_label,
        # img=imgviz.rgb2gray(image_0),
        font_size=15,
        # label_names=class_names,
        loc="rb", )
    imgviz.io.imsave(out_viz_file, viz)

    fg = pr_label > 0           # Obtain the foreground area
    jiaomo = pr_label == 5      # Obtain the jiaomo area
    pr_label[pr_label > 0] = 1
    # pr_label[jiaomo] = 5

    # print(type(fg))
    fg = np.uint8(fg)
    jiaomo = np.uint8(jiaomo)

    # getting the position of eye
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_c = max(contours, key=cv2.contourArea)

    # save image
    # cv2.drawContours(image, largest_c, -1, (0, 0, 255), 3)
    # cv2.imwrite(f'/home/ubuntu/wx_test/media/seg_roi/'+os.path.basename(imgname), fg)

    bounding_box_eye = cv2.boundingRect(largest_c)
    print(bounding_box_eye[0]*w0/512, bounding_box_eye[1]*h0/512,
      bounding_box_eye[2]*w0/512, bounding_box_eye[3]*h0/512)
    x, y, w, h = int(bounding_box_eye[0]*w0/512), int(bounding_box_eye[1]*h0/512), int(bounding_box_eye[2]*w0/512), int(bounding_box_eye[3]*h0/512)

    roi = image_0[y:y+h, x:x+w]

    return roi

if __name__ == "__main__":
    print(seg('/home/ubuntu/wx_test/media/img/OSoa86c5Iej51vGe79ocmB3tW4vkAQ1629264398.jpg'))
