import cv2
import numpy as np

from matplotlib import pyplot as plt

img1 = cv2.imread('j.jpg', 0)  # 查询图片  queryImage

img2 = cv2.imread('k.jpg', 0)  # 训练图片  trainImage

# 初始化SIFT探测器

orb = cv2.ORB_create()

# 用SIFT找到关键点和描述符h

kp1, des1 = orb.detectAndCompute(img1, None)

kp2, des2 = orb.detectAndCompute(img2, None)

'''

接下来创建一个距离测量值为cv2.NORM_HAMMING的BFMatcher对象

用Matcher.match()来获取两幅图像中的最佳匹配；

我们按照距离的升序对它们进行排序

'''

# 创建BFMatcher对象

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 匹配器描述符：结果是DMatch对象的列表，该DMatch对象具有以下属性：

# 1.DMatch.distance:描述符之间的距离。越低，它就越好。

# 2.DMatch.trainIndex:训练描述符中描述符的索引

# 3.DMatch.queryIndex:查询描述符中描述符的索引

# 4.DMatch.imgIndex:训练图像的索引

matches = bf.match(des1, des2)

# 根据距离排序,第六个参数是outImg

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

plt.imshow(img3)
plt.savefig('l.jpg')
plt.show()
