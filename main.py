import cv2
import numpy as np
from PIL import Image

# 生成权重数组
def wei():
    W = []
    for i in range(11):
        for j in range(11 - i):
            k = 10 - (i + j)
            W.append([i / 10.0, j / 10.0, k / 10.0])
    return W

# 计算颜色向量之差并添加到数组
def add(c0, c1, P, det):
    Rx = c0[2] / 255.0 - c1[2] / 255.0
    Gx = c0[1] / 255.0 - c1[1] / 255.0
    Bx = c0[0] / 255.0 - c1[0] / 255.0

    # 计算颜色差异的欧氏距离
    d = np.sqrt(Rx * Rx + Gx * Gx + Bx * Bx) / 1.41

    # 如果距离大于等于阈值，则添加到数组
    if d >= 0.05:
        P.append([Rx, Gx, Bx])
        det.append(d)

# RTC-Gray算法主函数
def rtcprgb2gray(im):
    # 计算缩放因子
    s = 64.0 / np.sqrt(float(im.shape[0] * im.shape[1]))
    cols = int(s * im.shape[1] + 0.5)
    rows = int(s * im.shape[0] + 0.5)

    # 参数设定
    sigma = 0.05
    W = wei()

    # 初始化存储颜色差异和距离的数组
    P = []
    det = []

    # 随机采样位置数组
    pos0 = []
    for i in range(cols):
        for j in range(rows):
            x = int((i + 0.5) * im.shape[1] / cols)
            y = int((j + 0.5) * im.shape[0] / rows)
            pos0.append((x, y))

    # 创建pos1数组并随机打乱
    pos1 = pos0.copy()
    np.random.shuffle(pos1)

    # 对每个位置对计算颜色差异并添加到数组
    for i in range(len(pos0)):
        c0 = im[pos0[i][1], pos0[i][0]]
        c1 = im[pos1[i][1], pos1[i][0]]

        add(c0, c1, P, det)

    # 缩小图像区域
    cols //= 2
    rows //= 2

    # 对图像的不同区域进行采样和差异计算
    for i in range(cols - 1):
        for j in range(rows):
            x0 = int((i + 0.5) * im.shape[1] / cols)
            x1 = int((i + 1.5) * im.shape[1] / cols)
            y = int((j + 0.5) * im.shape[0] / rows)

            c0 = im[y, x0]
            c1 = im[y, x1]

            add(c0, c1, P, det)

    for i in range(cols):
        for j in range(rows - 1):
            x = int((i + 0.5) * im.shape[1] / cols)
            y0 = int((j + 0.5) * im.shape[0] / rows)
            y1 = int((j + 1.5) * im.shape[0] / rows)

            c0 = im[y0, x]
            c1 = im[y1, x]

            add(c0, c1, P, det)

    # 计算最优权重
    maxEs = -float('inf')
    bw = 0
    for i in range(len(W)):
        Es = 0
        for j in range(len(P)):
            L = np.dot(P[j], W[i])
            detM = det[j]

            a = (L + detM) / sigma
            b = (L - detM) / sigma

            Es += np.log(np.exp(-a * a) + np.exp(-b * b))
        Es /= len(P)

        # 选择最大能量函数的权重
        if Es > maxEs:
            maxEs = Es
            bw = i

    # 分离图像通道
    c = cv2.split(im)

    # 根据最优权重合成去色图像
    result = cv2.addWeighted(c[2], W[bw][0], c[1], W[bw][1], 0.0)
    result = cv2.addWeighted(c[0], W[bw][2], result, 1.0, 0.0)

    return result



if __name__ == '__main__':

  origin_img = cv2.imread('./3.png')
  cv2.imshow('Original Image', origin_img)
  img = cv2.imread('./3.png', cv2.IMREAD_GRAYSCALE)

  # 直方图均衡化
  equalized_image = cv2.equalizeHist(img)
  cv2.imshow('Equalized Image', equalized_image)
  
  # 灰度图
  gray_image = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
  cv2.imshow('Grayscale Image', gray_image)

  # 对比图去色
  decolorized_image = rtcprgb2gray(origin_img)
  cv2.imshow('Decolorized Image', decolorized_image)

  #自适应直方图均衡化
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  clahe_image = clahe.apply(img)
  cv2.imshow('CLAHE Image', clahe_image)

  # 对比度拉伸
  min_val, max_val, _, _ = cv2.minMaxLoc(img)
  stretched_image = np.uint8(255 * (img - min_val) / (max_val - min_val))
  cv2.imshow('Stretched Image', stretched_image)

  # Gamma校正
  gamma = 1.5
  gamma_corrected_image = np.uint8(((img / 255.0) ** gamma) * 255)
  cv2.imshow('Gamma Corrected Image', gamma_corrected_image)

  # 对比图去色 + 直方图均衡化
  decolorized_equalized_image = cv2.equalizeHist(gray_image)
  cv2.imshow('Decolorized Equalizedected Image', decolorized_equalized_image)

  cv2.waitKey(0)
  cv2.destroyAllWindows()


