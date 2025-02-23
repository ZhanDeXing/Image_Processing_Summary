# %%
import cv2
# from colorspace import *
# from noises import *
# from filters import *
# from histeq import *
from retinex import *
from retinex.enhancer import *
from retinex.retinex_cv import SSR

# %%
img_bgr = cv2.imread("pictureTest/0.png")  # 读取图像文件
cv2.imshow('BGR', img_bgr)  # 显示原始图像

# img_ssr = SSR(img_bgr, 3)
# cv2.imshow('img_ssr', img_ssr)

# 多尺度Retinex
# %%
config = {
    "sigma_list": [15, 80, 250],
    "G": 5.0,
    "b": 25.0,
    "alpha": 125.0,
    "beta": 46.0,
    "low_clip": 0.01,
    "high_clip": 0.99
}

img_tmp = np.float64(img_bgr)
img_msr_logR = multi_scale_retinex(img_tmp, config['sigma_list'])
dst_R = cv2.normalize(img_msr_logR, None, 0, 255, cv2.NORM_MINMAX)     # 将归一化像素点变为0~255像素值
img_msr = cv2.convertScaleAbs(dst_R)  # 线性变换并取绝对值
cv2.imshow('img_msr', img_msr)

img_msrcr = MSRCR(
    img_bgr,
    config['sigma_list'],
    config['G'],
    config['b'],
    config['alpha'],
    config['beta'],
    config['low_clip'],
    config['high_clip']
)
cv2.imshow("MSRCR", img_msrcr)

img_amsrcr = automated_MSRCR(
        img_bgr,
        config['sigma_list']
    )
cv2.imshow("AMSRCR", img_amsrcr)    # 统计加权（auto

img_msrcp = MSRCP(
        img_bgr,
        config['sigma_list'],
        config['low_clip'],
        config['high_clip']
    )
cv2.imshow("MSRCP", img_msrcp)

sigma_list = [15, 80, 250]
# img_attnmsr = AttnMSR(img_bgr, sigma_list, 10)
# cv2.imshow("AttnMSR", img_attnmsr)
# img_mss = multi_scale_sharpen(img_attnmsr)
# cv2.imshow("AttnMSR+MSS", img_mss)

from matplotlib import pyplot as plt

# # 绘制直方图
# color = ('b', 'g', 'r')
# for i, cl in enumerate(color):
#     hist = cv2.calcHist([img_bgr], [i], None, [256], [0, 256])
#     plt.plot(hist, color=cl)
#     plt.xlim([0, 256])
# for i, cl in enumerate(color):
#     hist = cv2.calcHist([img_attnmsr], [i], None, [256], [0, 256])
#     plt.plot(hist, '--', color=cl)
#     plt.xlim([0, 256])
# plt.show()

# %%
cv2.waitKey()
# 保存图像操作
# %%
# cv2.imwrite("enlighten.png", np.concatenate((img_bgr, img_attnmsr, img_mss), axis=1))
