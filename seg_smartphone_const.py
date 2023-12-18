# coding: utf-8
import json
import os
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import torch
# from eye_dataset import Dataset, get_preprocessing
import albumentations as albu
import imgviz
from PIL import Image
import math


# 最大连通域
def find_max_region(mask_sel):
    contours, hierarchy = cv2.findContours(mask_sel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 找到最大区域并填充
    area = []

    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))

    max_idx = np.argmax(area)

    max_area = cv2.contourArea(contours[max_idx])

    for k in range(len(contours)):

        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)
    return mask_sel


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


def seg_cornea(Pr_label):
    viz = imgviz.label2rgb(
        label=Pr_label,
        # img=imgviz.rgb2gray(image_0),
        font_size=15,
        # label_names=class_names,
        loc="rb", )
    # 灰度
    Frame = cv2.cvtColor(viz, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    Frame = cv2.GaussianBlur(Frame, (5, 5), 0)
    # 二值化
    Ret, Frame = cv2.threshold(Frame, 50, 255, cv2.THRESH_BINARY)
    # 最大连通域
    Frame = find_max_region(Frame)
    return Frame


def seg_conectComponents(Frame):
    num_labels, label_s, stats, centroids = cv2.connectedComponentsWithStats(Frame, connectivity=8)
    # print(stats)  # 连通域的信息：对应各个轮廓的x、y、width、height和面积
    # print(centroids[1])  # 连通域的中心点
    # print(stats[1][3])
    minR = stats[1][3] / 2
    maxR = stats[1][2] / 2
    L_diameter = stats[1][3]
    H_diameter = stats[1][2]
    # 用横径计算比例
    min_ratio = 11 / stats[1][2]
    max_ratio = 12 / stats[1][2]
    ratio = (min_ratio + max_ratio) / 2
    if minR > maxR:
        temp = minR
        minR = maxR
        maxR = temp
    Contours, hierarchy = cv2.findContours(Frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return ratio, L_diameter, H_diameter, minR, maxR, min_ratio, max_ratio, centroids[1], stats[1][4], Contours


def model_pterygium(Model1, X_tensor, w0, h0):
    pr_mask = Model1.predict(X_tensor)
    Pr_label = torch.argmax(pr_mask, dim=1)
    Pr_label = Pr_label.squeeze().cpu().numpy()

    # fg = pr_label > 0  # Obtain the foreground area
    pterygium = Pr_label == 2  # Obtain the area
    Pr_label[Pr_label > 0] = 0
    Pr_label[pterygium] = 2

    pterygium = np.uint8(pterygium)
    pterygium = cv2.resize(pterygium, (w0, h0), interpolation=cv2.INTER_NEAREST)
    Contours, hierarchy = cv2.findContours(pterygium, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(len(Contours), Contours)
    contour = []
    for item in range(len(Contours)):
        # if cv2.contourArea(Contours[item]) > 3000:
        if cv2.contourArea(Contours[item]) > 3000:
            # print(cv2.contourArea(Contours[item]))
            contour.append(Contours[item])
    sort(contour)
    # for i in range(len(contour)):
    #     print(contour[i][0])
    return contour


def model_pterygium_mask(Model1, X_tensor, Frame, file_name, img_origin, w0, h0):
    pr_mask = Model1.predict(X_tensor)
    Pr_label = torch.argmax(pr_mask, dim=1)
    Pr_label = Pr_label.squeeze().cpu().numpy()
    Pr_label = np.uint8(Pr_label)
    # print(np.unique(Pr_label))
    # fg = pr_label > 0  # Obtain the foreground area
    pterygium = Pr_label == 2  # Obtain the area
    Pr_label[Pr_label > 0] = 0
    Pr_label[pterygium] = 2

    # pterygium = np.uint8(pterygium)
    # pterygium = cv2.resize(pterygium, (w0, h0), interpolation=cv2.INTER_NEAREST)
    Pr_label = cv2.resize(Pr_label, (w0, h0), interpolation=cv2.INTER_NEAREST)
    Frame = cv2.resize(Frame, (w0, h0), interpolation=cv2.INTER_NEAREST)
    Pterygium = imgviz.label2rgb(
        label=Pr_label,
        font_size=15,
        loc="rb", )
    image_mask = cv2.add(Pterygium, np.zeros(np.shape(Pterygium), dtype=np.uint8), mask=Frame)
    # image_mask = cv2.drawContours(image_mask, [Contours[0]], -1, (0, 255, 0), 1)
    # out_viz_file3 = os.path.join('./img_result20220120/20220120mask', case)
    # imgviz.io.imsave(out_viz_file3, image_mask)
    image_mask = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
    Ret, Image_mask = cv2.threshold(image_mask, 50, 255, cv2.THRESH_BINARY)
    out_viz_file3 = os.path.join('/home/ubuntu/wx_test/media/seg_mask/' + file_name)
    # 保存图片函数++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    imgviz.io.imsave(out_viz_file3, Image_mask)
    Contours, hierarchy = cv2.findContours(Image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    number = len(Contours)
    contour = []
    for item in range(number):
        # if cv2.contourArea(Contours[item]) > 3000:
        if cv2.contourArea(Contours[item]) > 3000:
            print(cv2.contourArea(Contours[item]))
            contour.append(Contours[item])
    # print(len(contour), contour)
    sort(contour)
    zeros = np.zeros(img_origin.shape, dtype=np.uint8)
    for k in range(len(contour)):
        zeros = cv2.fillPoly(zeros, [contour[k]], color=(0, 165, 255))

    alpha = 1
    beta = 0.3
    gamma = 0
    img_mask = cv2.addWeighted(img_origin, alpha, zeros, beta, gamma)
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)
    out_viz_file4 = os.path.join('/home/ubuntu/wx_test/media/seg_result/' + file_name)
    # 保存图片函数++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    imgviz.io.imsave(out_viz_file4, img_mask)
    return contour, len(contour), img_mask


def model_cornea(Model, X_tensor):
    pr_mask = Model.predict(X_tensor)
    Pr_label = torch.argmax(pr_mask, dim=1)
    Pr_label = Pr_label.squeeze().cpu().numpy()
    # fg = pr_label > 0  # Obtain the foreground area
    jiaomo = Pr_label == 5  # Obtain the jiaomo area
    Pr_label[Pr_label > 0] = 0
    Pr_label[jiaomo] = 5
    return Pr_label


def judge(label):
    flag = False
    for i in label:
        if i == 2:
            flag = True
    return flag


def move_by(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    m1 = (dx * dx + dy * dy) ** 0.5
    return m1


def sort(contour):
    if len(contour) > 1:
        # if contour[0][0][0][0] > contour[1][0][0][0]:
        #     contour[0], contour[1] = contour[1], contour[0]
        for j in range(0, len(contour) - 1):
            count = j
            for i in range(j, len(contour) - 1):
                if contour[count][0][0][0] > contour[i + 1][0][0][0]:
                    count = i + 1
            if count != j:
                contour[j], contour[count] = contour[count], contour[j]


def Computer_DIST(points, number):
    Max = 0
    for i in range(0, number - 1):
        for j in range(i + 1, number):
            temp = move_by(points[i][0], points[i][1], points[j][0], points[j][1])
            if temp > Max:
                Max = temp
    return Max


def Fitting_Ellipse(Frame, img_origin):
    # imgray = cv2.Canny(Frame, 600, 100, 3)  # Canny边缘检测，参数可更改
    # # cv2.imshow("0",imgray)
    # ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
    Contours, hierarchy = cv2.findContours(Frame, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)  # contours为轮廓集，可以计算轮廓的长度、面积等
    # print("Coutours:", Contours, len(Contours))
    Mask = np.zeros(Frame.shape[:2], dtype=np.uint8)
    for cnt in Contours:
        if len(cnt) > 50:
            S1 = cv2.contourArea(cnt)
            # print("s1:", S1)

            ell = cv2.fitEllipse(cnt)
            S2 = math.pi * ell[1][0] * ell[1][1]
            # print("s2:", S2)
            if (S1 / S2) > 0.2:  # 面积比例，可以更改，根据数据集。。。
                Mask = cv2.ellipse(Mask, ell, (255, 255, 255), -1)

    return Mask


def LookPoint(Center, Point, Cts):
    Distance = cv2.pointPolygonTest(Cts, Center, False)
    # print(Distance)
    return Distance
    # pt1 = Cts[0][0][0]
    # if pt1 < Center[0] < Point[0] or pt1 > Center[0] > Point[0]:
    #     return True
    # else:
    #     return False


def seg(img, imgname):
    print("Test!!!!!!!!!")
    List = []
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax2d'  # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cpu'

    #
    model = torch.load('/home/ubuntu/wx_test/segmentation/weights/seg_best_model.pth', map_location='cpu')  # 角膜分割模型（拟合圆）
    # model1 = torch.load('./weights/Phone_Seg_Cornea_Ptery_20220404.pth')  # 胬肉分割模型
    model1 = torch.load('/home/ubuntu/wx_test/segmentation/weights/Seg_Ptery_SL1438_Phone118_20220531.pth', map_location='cpu')  # 胬肉分割模型

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preprocessing = get_preprocessing(preprocessing_fn)

    file_name = os.path.basename(imgname)
    image_0 = cv2.imread(img)
    print(file_name)
    # 分割
    image_original = image_0
    h0, w0 = image_0.shape[0:2]
    image = cv2.resize(image_0, (512, 512))
    img = preprocessing(image=image)['image']
    x_tensor = torch.from_numpy(img).to(DEVICE).unsqueeze(0)
    Contours_pterygium = model_pterygium(model1, x_tensor, w0, h0)  # 获取胬肉的轮廓
    # 获取角膜的轮廓
    pr_label = model_cornea(model, x_tensor)
    pr_label = cv2.resize(pr_label, (w0, h0), interpolation=cv2.INTER_NEAREST)
    mask = np.zeros(image_original.shape[:2], dtype=np.uint8)
    frame = seg_cornea(pr_label)
    # 求cornea的各种参数：标准比例，纵径，横径，最小半径，最大半径，最小比例，最大比例，中心点坐标，cornea区域面积，cornea轮廓
    frame = Fitting_Ellipse(frame, image_original)  # 最小二乘法拟合椭圆
    ratio, l_diameter, h_diameter, minR, maxR, min_ratio, max_ratio, center, cornea_area, Contours_cornea = seg_conectComponents(
        frame)
    # 求pterygium的各种参数：胬肉轮廓数组[]，胬肉轮廓的数量(注意！！！这里是被mask的胬肉轮廓)
    contours, numbers, img_masked = model_pterygium_mask(model1, x_tensor, frame, file_name, image_original, w0, h0)

    Pr_mask = model1.predict(x_tensor)
    labels = torch.argmax(Pr_mask, dim=1)
    labels = labels.squeeze().cpu().numpy()
    labels = np.uint8(labels)
    # print(np.unique(labels))
    # 判断胬肉返回是否为空
    Flag = judge(np.unique(labels))
    # print(Flag)
    # 开始计算指标
    if not Flag or numbers == 0:
        return List
    else:

        # for contour in contours:
        # print(numbers)
        for i in range(numbers):
            Dict = {'Width': 0, 'Length': 0, 'Area': 0, 'Area_p': 0}
            # 求width---------------------------------------------------------------------------------------------
            pts = []
            # print(Contours_cornea[0])
            for pt in Contours_pterygium[i]:
                # print(contours[i])
                # print(pt[0])
                point_x = int(pt[0][0])
                point_y = int(pt[0][1])
                distance = cv2.pointPolygonTest(Contours_cornea[0], (int(point_x), int(point_y)), True)
                if abs(distance) <= 2:
                    pts.append((point_x, point_y))
                    # print(pt[0], distance)
            # print(len(pts), pts)
            # print(pts[0][0])
            width = Computer_DIST(pts, len(pts))
            # min_width = width * min_ratio
            # max_width = width * max_ratio
            width = width * max_ratio
            print("胬肉 %d 侵入宽度width为: %.3f mm" % (i + 1, width))
            Dict['Width'] = width
            # 求length---------------------------------------------------------------------------------------------
            min_num = float(maxR)
            # 定义最近的点pt_closed
            pt_closed = center
            for point in contours[i]:
                # print(point[0][0], point[0][1])
                m = move_by(center[0], center[1], point[0][0], point[0][1])
                if m < min_num:
                    min_num = m
                    pt_closed = point[0]
            # if not LookPoint(center, pt_closed, contours[i]):
            if LookPoint(center, pt_closed, contours[i]) < 0:
                min_length = (float(minR) - min_num)
                max_length = (float(maxR) - min_num)
                min_invasion = min_length * min_ratio
                max_invasion = max_length * max_ratio
                length = max_length * max_ratio
            else:
                min_length = (float(minR) + min_num)
                max_length = (float(maxR) + min_num)
                min_invasion = min_length * min_ratio
                max_invasion = max_length * max_ratio
                length = max_length * max_ratio
            # radius = h_diameter / 2
            # invasion_percentage = (radius - min_num) / radius
            # print("radius, min_num, invasion_percentage:", radius, min_num, invasion_percentage)
            # print("翼状胬肉侵入比值为：%.6f " % invasion_percentage)
            print("胬肉 %d 侵入距离Length是: %.3f mm" % (i + 1, length))
            Dict['Length'] = length
            # 求area---------------------------------------------------------------------------------------------
            area = cv2.contourArea(contours[i])
            temp = area / cornea_area
            AreaP_min = 3.14 * minR * min_ratio * minR * min_ratio * temp
            AreaP_max = 3.14 * maxR * max_ratio * maxR * max_ratio * temp
            AreaP = 3.14 * maxR * max_ratio * maxR * max_ratio * temp
            # print("胬肉", i, "：")
            # print("面积：", stats[i][4])
            print("胬肉 %d 侵入角膜像素面积Area是:%d" % (i + 1, area))
            print("胬肉 %d 侵入角膜实际面积Area是:%.3f mm²" % (i + 1, AreaP))
            print("胬肉 %d 侵入角膜面积占整个角膜面积的百分比是: %.3f %%" % (i + 1, temp * 100))
            Dict['Area'] = AreaP
            Dict['Area_p'] = temp * 100
            List.append(Dict)
        # return List, img_masked
        print('整个角膜像素面积为：', cornea_area)
        print('整个角膜实际面积为：', 3.14 * maxR * max_ratio * maxR * max_ratio)
        return List


if __name__ == '__main__':
    print(seg('/home/ubuntu/wx_test/segmentation/Wangxiuli.jpg', 'Wangxiuli.jpg'))
