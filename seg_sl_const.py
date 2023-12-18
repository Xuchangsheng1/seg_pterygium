import json
import random
from mmdet.apis import init_detector, inference_detector
import mmcv
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
import csv

# class_names = ['_background_', 'palpebra superior', 'palpebra inferior',
#                'carunculae lacrimalis', 'conjunctiva', 'cornea']
class_names = ['backgoroud', 'cornea', 'pterygium']


def make_print_to_file(path='./'):
    """
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    """
    import sys
    import os
    import config_file as cfg_file
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('SL_model0531_segMain-Day' + '%Y_%m_%d' + '_Time' + '%H:%M:%S')
    sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60, '*'))


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
    viz_file = os.path.join('/home/eye/XCS/File/220707_slit_vs_smart/slit_result/result_cornea', file_name1 + '_cornea.jpg')
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
    # 保存图片函数++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    imgviz.io.imsave(viz_file, Frame)
    # print(viz_file)
    return Frame


def seg_hough(Frame):
    Frame = cv2.cvtColor(Frame, cv2.COLOR_GRAY2BGR)
    # 灰度化
    gray = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    # 输出图像大小，方便根据图像大小调节minRadius和maxRadius
    # print(viz.shape)
    # 霍夫变换圆检测
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 14, 1500, param1=83, param2=30, minRadius=10, maxRadius=700)
    # 判断
    if circles is None:
        # print(circles)
        out_viz_file2 = os.path.join('./img_result20220120/20220120wrong', file_name1 + '_wrong.jpg')
        # 保存图片函数++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # imgviz.io.imsave(out_viz_file2, image_0)

    else:
        # print(circles)
        # 输出检测到圆的个数
        # print(len(circles[0]))

        # print('-------------我是条分割线-----------------')
        # 根据检测到圆的信息，画出每一个圆
        for circle in circles[0]:
            # 圆的基本信息
            # print(circle[2])
            # 坐标行列
            x = int(circle[0])
            y = int(circle[1])
            # 半径
            r = int(circle[2])
            # print(x, y, r)
            # 在原图用指定颜色标记出圆的位置
            image_hough = cv2.circle(image_original, (x, y), r, (255, 0, 0), 2)
            image_hough = cv2.cvtColor(image_hough, cv2.COLOR_BGR2RGB)
            # mask操作
            mask = np.zeros(image_original.shape[:2], dtype=np.uint8)
            mask = cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
            image_mask = cv2.add(image_original, np.zeros(np.shape(image_original), dtype=np.uint8), mask=mask)
            image_mask = cv2.cvtColor(image_mask, cv2.COLOR_BGR2RGB)
            out_viz_file3 = os.path.join('./img_result20220120/20220120mask', file_name1 + '_mask.jpg')
            # 保存图片函数++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # imgviz.io.imsave(out_viz_file3, image_mask)
            # 显示新图像
            # cv2.imshow('res', img_hough)
            out_viz_file1 = os.path.join('./img_result20220120/20220120hough', file_name1 + '_hough.jpg')
            # 保存图片函数++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # imgviz.io.imsave(out_viz_file1, image_hough)
            return x, y, r


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


def model_pterygium(Model1, X_tensor):
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
        # if cv2.contourArea(Contours[item]) > 9000:
        if cv2.contourArea(Contours[item]) > 500:
            # print(cv2.contourArea(Contours[item]))
            contour.append(Contours[item])
    sort(contour)
    # for i in range(len(contour)):
    #     print(contour[i][0])
    return contour


def model_pterygium_mask(Model1, X_tensor, Frame, cornea_S):
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
    out_viz_file2 = os.path.join('/home/eye/XCS/File/220707_slit_vs_smart/slit_result/result_pterygium', file_name1 + '_result_pterygium.jpg')
    Pterygium = imgviz.label2rgb(
        label=Pr_label,
        font_size=15,
        loc="rb", )
    # 保存图片函数++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    imgviz.io.imsave(out_viz_file2, Pterygium)
    image_mask = cv2.add(Pterygium, np.zeros(np.shape(Pterygium), dtype=np.uint8), mask=Frame)
    # image_mask = cv2.drawContours(image_mask, [Contours[0]], -1, (0, 255, 0), 1)
    # out_viz_file3 = os.path.join('./img_result20220120/20220120mask', case)
    # imgviz.io.imsave(out_viz_file3, image_mask)
    image_mask = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
    Ret, Image_mask = cv2.threshold(image_mask, 50, 255, cv2.THRESH_BINARY)
    out_viz_file3 = os.path.join('/home/eye/XCS/File/220707_slit_vs_smart/slit_result/result_mask', file_name1 + '_result_mask.jpg')
    # 保存图片函数++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    imgviz.io.imsave(out_viz_file3, Image_mask)
    Contours, hierarchy = cv2.findContours(Image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    number = len(Contours)
    contour = []
    for item in range(number):
        # if cv2.contourArea(Contours[item]) > 9000:
        if cv2.contourArea(Contours[item]) > 500:
            print(cv2.contourArea(Contours[item]))
            contour.append(Contours[item])
    # print(len(contour), contour)
    sort(contour)
    return contour, len(contour)


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


def Overlap_Rate(contour1, contour2, Mask, Num):
    # print(contour1, type(contour1))
    # print(contour2, type(contour2))
    Area1 = cv2.contourArea(contour1[Num])
    Area2 = cv2.contourArea(contour2)
    # 处理轮廓1
    mask1 = np.zeros(Mask.shape[:2], dtype=np.uint8)
    points1 = []
    # print(contour2)
    for pt1 in contour1[Num]:
        point_X = int(pt1[0][0])
        point_Y = int(pt1[0][1])
        points1.append((point_X, point_Y))
    # print(points1)
    color = [255, 255, 255]
    cv2.fillPoly(mask1, [np.array(points1)], color)
    # img1 = cv2.drawContours(Mask, contour1, 0, (255, 255, 255), -1)
    out_viz_file1 = os.path.join('/home/eye/XCS/File/220707_slit_vs_smart/slit_result/ptery_origin', file_name1 + str(Num) + '_ptery_origin.jpg')
    imgviz.io.imsave(out_viz_file1, mask1)
    # 处理轮廓2
    contour2 = np.trunc(contour2).astype(int).tolist()
    contour2 = np.array(contour2)
    contour2 = contour2.astype(np.int32)
    points2 = []
    # print(contour2)
    for pt2 in contour2:
        point_X = int(pt2[0][0])
        point_Y = int(pt2[0][1])
        points2.append((point_X, point_Y))
    # print(points2)
    # img2 = cv2.drawContours(Mask, contour2, 0, (255, 255, 255), -1)
    color = [255, 255, 255]
    cv2.fillPoly(Mask, [np.array(points2)], color)
    out_viz_file2 = os.path.join('/home/eye/XCS/File/220707_slit_vs_smart/slit_result/ptery_label', file_name1 + '_ptery_label.jpg')
    imgviz.io.imsave(out_viz_file2, Mask)

    image_interact = cv2.add(mask1, np.zeros(np.shape(mask1), dtype=np.uint8), mask=Mask)
    out_viz_file3 = os.path.join('/home/eye/XCS/File/220707_slit_vs_smart/slit_result/img_interact', file_name1 + str(Num) + '_img_interact.jpg')
    imgviz.io.imsave(out_viz_file3, image_interact)
    Contours, hierarchy = cv2.findContours(image_interact, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if Num == 3:
        return 0
    # elif not Contours:
    #     print("模型推理错误：多出现胬肉")
    #     return Overlap_Rate(contour1, contour2, Mask, Num + 1)
    else:
        # print(Contours)
        Area_interact = 0
        for cts in Contours:
            tem = cv2.contourArea(cts)
            if tem > Area_interact:
                Area_interact = tem
        print("重叠的面积部分：", Area_interact)
        OverlapRate = Area_interact / (Area2 + Area1 - Area_interact)
        Acc = 1 - (Area2 + Area1 - 2 * Area_interact) / 2000768
        global miou, mPrecision, mRecall, mAcc
        miou += OverlapRate
        Precision = Area_interact / Area1
        Recall = Area_interact / Area2
        mPrecision += Precision
        mRecall += Recall
        mAcc += Acc
        print("Area IoU：", OverlapRate)
        print("Area Precision：", Precision)
        print("Area Recall：", Recall)
        print("Area Acc：", Acc)
        return Area2 - Area1


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
                Mask_hough = cv2.ellipse(img_origin, ell, (255, 0, 0), 6)
                img_mask = cv2.add(img_origin, np.zeros(np.shape(img_origin), dtype=np.uint8), mask=Mask)
                img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)
                Mask_hough = cv2.cvtColor(Mask_hough, cv2.COLOR_BGR2RGB)
                out_viz_file5 = os.path.join('/home/eye/XCS/File/220707_slit_vs_smart/slit_result/result_cornea_mask',
                                             file_name1 + '_result_cornea_mask.jpg')
                # 保存图片函数++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                imgviz.io.imsave(out_viz_file5, img_mask)
                out_viz_file4 = os.path.join('/home/eye/XCS/File/220707_slit_vs_smart/slit_result/result_cornea_hough',
                                             file_name1 + '_result_cornea_hough.jpg')
                # 保存图片函数++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                imgviz.io.imsave(out_viz_file4, Mask_hough)
    out_viz_file3 = os.path.join('/home/eye/XCS/File/220707_slit_vs_smart/slit_result/result_cornea_fitting',
                                 file_name1 + '_result_cornea_fitting.jpg')
    # 保存图片函数++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    imgviz.io.imsave(out_viz_file3, Mask)

    return Mask


def get_distance_from_point_to_line(point, line_point1, line_point2):
    # 对于两点坐标为同一点时,返回点与点的距离
    if line_point1 == line_point2:
        point_array = np.array(point)
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array - point1_array)
    # 计算直线的三个参数
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]
    # 根据点到直线的距离公式计算距离
    Distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A ** 2 + B ** 2))
    return Distance


def Computer_loss(art, min, max):
    if min < art < max or min == max:
        return 1
    elif art < min:
        loss = 1 - (min - art) / art
        return loss
    else:
        loss = 1 - (art - max) / art
        return loss


def Computer_loss_const(art, const):
    loss = 1 - abs(const - art) / art
    return loss


def Computer_ROC(art, min, max):
    if 0 <= art < 2:
        true = 2
    elif 2 <= art < 4:
        true = 4
    else:
        true = 6

    if 0 <= max <= 2:
        value1 = value2 = 1
        predict1 = predict2 = 3
    elif min <= 2 < max:
        value1 = abs(max - 2) / (max - min)
        value2 = abs(min - 2) / (max - min)
        predict1 = 5
        predict2 = 3
    elif 2 < min < max <= 4:
        value1 = 1
        value2 = 1
        predict1 = 5
        predict2 = 5
    elif min <= 4 < max:
        value1 = abs(max - 4) / (max - min)
        value2 = abs(min - 4) / (max - min)
        predict1 = 7
        predict2 = 5
    else:
        value1 = 1
        value2 = 1
        predict1 = 7
        predict2 = 7

    return true, predict1, predict2, value1, value2


def Computer_ROC_const(wid_art, wid_pre, len_art, len_pre, area_art, area_pre):  # 输入人工标记的值和测量出来的值
    global T_1, T_2, T_3, P_2, P_1, P_3, True_Pre, False_Pre
    if (wid_art >= 5 and area_art >= 6.25) or len_art >= 4:
        true = 6
        T_3 += 1
    elif 2 <= len_art < 4:
        true = 4
        T_2 += 1
    else:
        true = 2
        T_1 += 1

    if (wid_pre >= 5 and area_pre >= 6.25) or len_pre >= 4:
        predict = 7
        T_3 += 1
    elif 2 <= len_pre < 4:
        predict = 5
        T_2 += 1
    else:
        predict = 3
        T_1 += 1
    if true + 1 == predict:
        print("预测正确")
        True_Pre += 1
    else:
        print("预测错误")
        False_Pre += 1

    return true, predict


def LookPoint(Center, Point, Cts):
    print(Center)
    point_X = int(Center[0])
    point_Y = int(Center[1])
    Distance = cv2.pointPolygonTest(Cts, (int(point_X), int(point_Y)), False)
    # print(Distance)
    return Distance
    # pt1 = Cts[0][0][0]
    # if pt1 < Center[0] < Point[0] or pt1 > Center[0] > Point[0]:
    #     return True
    # else:
    #     return False


if __name__ == '__main__':

    make_print_to_file(path='/home/eye/XCS/segmentation/Log')
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax2d'  # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cuda'
    TP_width = 0
    TN_width = 0
    FP_width = 0
    FN_width = 0
    TP_length = 0
    TN_length = 0
    FP_length = 0
    FN_length = 0
    True_width = 0
    False_width = 0
    True_length = 0
    False_length = 0
    True_Pre = 0
    False_Pre = 0
    mark = 0
    miou = 0
    mHausdorffDistance = 0
    mPrecision = 0
    mRecall = 0
    mAcc = 0

    mWidth = 0
    mLength = 0
    mArea = 0

    T_0 = 0
    T_1 = 0
    T_2 = 0
    T_3 = 0
    P_0 = 0
    P_1 = 0
    P_2 = 0
    P_3 = 0

    head = [['r0', '0', 'r1', '1', 'r2', '2', 'r3', '3']]
    # head = [['r0', '0', 'r1', '1', 'r2', '2', 'r3', '3']]
    #
    model = torch.load('./weights/seg_best_model.pth')  # 角膜分割模型（拟合圆）
    model1 = torch.load('/home/eye/XCS/aieyes/segmentation/weights/Seg_Ptery_SL1438_20220526.pth')  # 胬肉分割模型
    # model1 = torch.load('./weights/best_model2.pth')  # 胬肉分割模型

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preprocessing = get_preprocessing(preprocessing_fn)

    # patients = os.listdir('/home/eye/XCS/File/220420Test')
    patients = os.listdir('/home/eye/XCS/File/220707_slit_vs_smart/slit_test')
    # patients = os.listdir('./img_test')
    for case in patients:
        file_name = os.path.splitext(case)[0]
        file_name1 = file_name
        print(file_name)
        file_name = f'{"/home/eye/XCS/File/220707_slit_vs_smart/slit_TestsetValue"}/{file_name + ".json"}'
        # print(file_name)
        # image_0 = cv2.imread(os.path.join('/home/eye/XCS/File/220420Test', case))
        image_0 = cv2.imread(os.path.join('/home/eye/XCS/File/220707_slit_vs_smart/slit_test', case))
        # image_0 = cv2.imread(os.path.join('./img_test', case))
        # 分割
        image_original = image_0
        h0, w0 = image_0.shape[0:2]
        image = cv2.resize(image_0, (512, 512))
        img = preprocessing(image=image)['image']
        x_tensor = torch.from_numpy(img).to(DEVICE).unsqueeze(0)
        Contours_pterygium = model_pterygium(model1, x_tensor)  # 获取胬肉的轮廓
        # 获取角膜的轮廓
        pr_label = model_cornea(model, x_tensor)
        pr_label = cv2.resize(pr_label, (w0, h0), interpolation=cv2.INTER_NEAREST)
        mask = np.zeros(image_original.shape[:2], dtype=np.uint8)
        frame = seg_cornea(pr_label)
        # 求cornea的各种参数：标准比例，纵径，横径，最小半径，最大半径，最小比例，最大比例，中心点坐标，cornea区域面积，cornea轮廓
        frame = Fitting_Ellipse(frame, image_original)  # 最小二乘法拟合椭圆
        ratio, l_diameter, h_diameter, minR, maxR, min_ratio, max_ratio, center, cornea_area, Contours_cornea = seg_conectComponents(frame)
        # 求pterygium的各种参数：胬肉轮廓数组[]，胬肉轮廓的数量(注意！！！这里是被mask的胬肉轮廓)
        contours, numbers = model_pterygium_mask(model1, x_tensor, frame, cornea_area)

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
            print("-----no invasion")
            d_data = [1, 1, 0, 0, 0, 0, 0, 0]
            T_0 += 1
            P_0 += 1
            head.append(d_data)
        else:

            # for contour in contours:
            print(numbers)
            for i in range(numbers):
                d_data = [0, 0, 0, 0, 0, 0, 0, 0]
                LengthP = 0
                R = (minR + maxR) / 2
                # ratio = (min_ratio + max_ratio) / 2
                number = str(i + 1)
                temp = str(i + 2)
                flag = False
                with open(file_name, "r") as f:
                    data = json.load(f)
                    width_test = data["width" + number]
                    length_test = data["length" + number]
                    Area_test = data["Area" + number]
                    contours_test = data["contours" + number]
                    if i == 0:
                        temp1 = data["contours" + temp]
                        contours_test = np.array(contours_test)
                        contours_test = contours_test.astype(np.float32)
                        # print(contours_test)
                        if not len(temp1) and len(contours_test):
                            if contours_test[0][0][0] > center[0]:
                                print(contours_test[0][0][0], center[0])
                                flag = True
                    # print(temp1)
                    contours_test = np.array(contours_test)
                    contours_test = contours_test.astype(np.float32)

                if not len(contours_test):
                    print("无标记，模型推理错误：多胬肉")

                    d_data[0] = 1
                    d_data[1] = 1
                    head.append(d_data)
                    break
                elif numbers == 2 and flag:
                    print("模型推理错误：左侧多出现胬肉，右侧正常")
                    i = i + 1
                    mark = mark + 1
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
                    width = width * min_ratio
                    print("胬肉 %d 侵入宽度width为: %.3f mm" % (i, width))
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
                    print("胬肉 %d 侵入距离Length是: %.3f mm" % (i, length))
                    # 求area---------------------------------------------------------------------------------------------
                    area = cv2.contourArea(contours[i])
                    temp = area / cornea_area
                    AreaP_min = 3.14 * minR * min_ratio * minR * min_ratio * temp
                    AreaP_max = 3.14 * maxR * max_ratio * maxR * max_ratio * temp
                    AreaP = 3.14 * maxR * min_ratio * maxR * min_ratio * temp
                    # print("胬肉", i, "：")
                    # print("面积：", stats[i][4])
                    print("胬肉 %d 侵入面积像素Area是:%d" % (i, area))
                    print("胬肉 %d 侵入面积Area是:%.3f mm²" % (i, AreaP))
                    print("胬肉 %d 侵入面积百分比是: %.3f %%" % (i, temp * 100))

                    # 计算HausdorffDistance
                    # 1.创建计算距离对象
                    hausdorff_sd = cv2.createHausdorffDistanceExtractor()
                    # 2.计算轮廓之间的距离
                    LOSS_Dis = hausdorff_sd.computeDistance(contours[i], contours_test)
                    mHausdorffDistance = mHausdorffDistance + LOSS_Dis * min_ratio
                    OverlapRate_Area = Overlap_Rate(contours, contours_test, mask, i)
                    # miou = miou + iou
                    LOSS_width = abs(width_test * min_ratio - width)
                    LOSS_length = abs(
                        length_test * min_ratio - length)
                    print("Loss-Distance\t LOSS_width=", LOSS_width)
                    print("Loss-Distance\t LOSS_length=", LOSS_length)

                    print("hausdorff-Distance\t LOSS=", LOSS_Dis * min_ratio, "mm")
                    print("OverlapArea(面积差值)=", abs(OverlapRate_Area * min_ratio * min_ratio), "mm²")

                    # print("width_test(人工标记测试出来的宽度width):", width_test * min_ratio, "~", width_test * max_ratio,
                    #       "mm")
                    # print("length_test(人工标记测试出来的长度length):", length_test * min_ratio, "~", length_test * max_ratio,
                    #       "mm")
                    width_art = width_test * min_ratio
                    length_art = length_test * min_ratio
                    # invasion_percentage_art = length_test / radius
                    # print("length_test, radius, invasion_percentage_art:", length_test, radius, invasion_percentage_art)
                    print("人工标记测试出来的宽度width: %.3f mm" % width_art)
                    print("人工标记测试出来的长度length: %.3f mm" % length_art)
                    temp1 = Area_test / cornea_area
                    AreaP_art = 3.14 * maxR * maxR * min_ratio * min_ratio * temp1
                    print("人工标记测试出来的面积Area:", AreaP_art, "mm²")
                    print("人工标记面积百分比是: %.3f %%" % (temp1 * 100))
                    width_temp = Computer_loss_const(width_art, width)
                    mWidth += width_temp
                    length_temp = Computer_loss_const(length_art, length)
                    mLength += length_temp
                    area_temp = Computer_loss_const(AreaP_art, AreaP)
                    mArea += area_temp
                    print("widthAccuracy,lengthAccuracy,areaAccuracy:", width_temp, length_temp, area_temp)
                    # t, p1, p2, v1, v2 = Computer_ROC(length_art, min_invasion, max_invasion)
                    # d_data[t] = 1
                    # d_data[p1] = v1
                    # d_data[p2] = v2
                    t, p = Computer_ROC_const(width_art, width, length_art, length, AreaP_art, AreaP)
                    d_data[t] = 1
                    d_data[p] = 1
                    head.append(d_data)
                    print(d_data)
                    break
                else:
                    mark = mark + 1
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
                    width = width * min_ratio
                    print("胬肉 %d 侵入宽度width为: %.3f mm" % (i + 1, width))
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
                    # 求area---------------------------------------------------------------------------------------------
                    area = cv2.contourArea(contours[i])
                    temp = area / cornea_area
                    AreaP_min = 3.14 * minR * min_ratio * minR * min_ratio * temp
                    AreaP_max = 3.14 * maxR * max_ratio * maxR * max_ratio * temp
                    AreaP = 3.14 * maxR * min_ratio * maxR * min_ratio * temp
                    # print("胬肉", i, "：")
                    # print("面积：", stats[i][4])
                    print("胬肉 %d 侵入面积像素Area是:%d" % (i + 1, area))
                    print("胬肉 %d 侵入面积Area是:%.3f mm²" % (i + 1, AreaP))
                    print("胬肉 %d 侵入面积百分比是: %.3f %%" % (i + 1, temp * 100))

                    # 计算HausdorffDistance
                    # 1.创建计算距离对象
                    hausdorff_sd = cv2.createHausdorffDistanceExtractor()
                    # 2.计算轮廓之间的距离
                    LOSS_Dis = hausdorff_sd.computeDistance(contours[i], contours_test)
                    mHausdorffDistance = mHausdorffDistance + LOSS_Dis * min_ratio
                    OverlapRate_Area = Overlap_Rate(contours, contours_test, mask, i)
                    # miou = miou + iou
                    LOSS_width = abs(width_test * min_ratio - width)
                    LOSS_length = abs(
                        length_test * min_ratio - length)
                    print("Loss-Distance\t LOSS_width=", LOSS_width)
                    print("Loss-Distance\t LOSS_length=", LOSS_length)

                    print("hausdorff-Distance\t LOSS=", LOSS_Dis * min_ratio, "mm")
                    print("OverlapArea(面积差值)=", abs(OverlapRate_Area * min_ratio * min_ratio), "mm²")

                    # print("width_test(人工标记测试出来的宽度width):", width_test * min_ratio, "~", width_test * max_ratio,
                    #       "mm")
                    # print("length_test(人工标记测试出来的长度length):", length_test * min_ratio, "~", length_test * max_ratio,
                    #       "mm")
                    width_art = width_test * min_ratio
                    length_art = length_test * min_ratio
                    # invasion_percentage_art = length_test / radius
                    # print("length_test, radius, invasion_percentage_art:", length_test, radius, invasion_percentage_art)
                    print("人工标记测试出来的宽度width: %.3f mm" % width_art)
                    print("人工标记测试出来的长度length: %.3f mm" % length_art)
                    temp1 = Area_test / cornea_area
                    AreaP_art = 3.14 * maxR * maxR * min_ratio * min_ratio * temp1
                    print("人工标记测试出来的面积Area:", AreaP_art, "mm²")
                    print("人工标记面积百分比是: %.3f %%" % (temp1 * 100))
                    width_temp = Computer_loss_const(width_art, width)
                    mWidth += width_temp
                    length_temp = Computer_loss_const(length_art, length)
                    mLength += length_temp
                    area_temp = Computer_loss_const(AreaP_art, AreaP)
                    mArea += area_temp
                    print("widthAccuracy,lengthAccuracy,areaAccuracy:", width_temp, length_temp, area_temp)
                    # t, p1, p2, v1, v2 = Computer_ROC(length_art, min_invasion, max_invasion)
                    # d_data[t] = 1
                    # d_data[p1] = v1
                    # d_data[p2] = v2
                    t, p = Computer_ROC_const(width_art, width, length_art, length, AreaP_art, AreaP)
                    d_data[t] = 1
                    d_data[p] = 1
                    head.append(d_data)
                    print(d_data)
                # LengthP = LengthP / 3
                # if LengthP > 1:
                #     print("需要做手术的概率是: 100 %")
                # elif LengthP == 0:
                #     print("需要做手术的概率是: 0.433 %")
                # else:
                #     print("需要做手术的概率是: %.3f %%" % (LengthP * 100))
        print('-------------我是条分割线-----------------')
    miou = miou / mark
    mHausdorffDistance = mHausdorffDistance / mark
    mPrecision = mPrecision / mark
    mRecall = mRecall / mark
    mAcc = mAcc / mark
    mWidth = mWidth / mark
    mLength = mLength / mark
    mArea = mArea / mark
    Acc_clc = True_Pre / (True_Pre + False_Pre)
    print("img number:", mark)
    print("miou:", miou)
    print("mPrecision", mPrecision)
    print("mRecall", mRecall)
    print("mAcc", mAcc)
    print("mHausdorffDistance", mHausdorffDistance, "mm")
    print("mWidth", mWidth)
    print("mLength", mLength)
    print("mArea", mArea)
    print("Acc_clc:", Acc_clc)

    with open("/home/eye/XCS/File/220707_slit_vs_smart/csv_File/220707_SL_3.csv", "w", encoding="utf-8", newline="") as f:
        csvf = csv.writer(f)
        for row in head:
            csvf.writerow(row)
    # Accuracy_width = True_width / (True_width + False_width)
    # Accuracy_length = True_length / (True_length + False_length)
    # print("长度正确率：", Accuracy_length, "宽度正确率：", Accuracy_width)
    # true negative rate，描述识别出的负例占所有负例的比例,TNR即为特异度（specificity）
    # TNR_width = TN_width / (FP_width + TN_width)
    # # true positive rate，召回率（Recall），也称查全率。描述识别出的所有正例占所有正例的比例,TPR即为敏感度（sensitivity）。
    # TPR_width = TP_width / (TP_width + FN_width)
    # # false positive rate，描述将负例识别为正例的情况占所有负例的比例
    # FPR_width = FP_width / (FP_width + TN_width)
    # # Positive predictive value，精确率（precision），也称为查准率，
    # PPV_width = TP_width / (TP_width + FP_width)
    # # classification accuracy，描述分类器的分类准确率
    # ACC_width = (TP_width + TN_width) / (TP_width + FP_width + FN_width + TN_width)
    # # balanced error rate
    # BER_width = 1 / 2 * (FPR_width + FN_width / (FN_width + TP_width))
