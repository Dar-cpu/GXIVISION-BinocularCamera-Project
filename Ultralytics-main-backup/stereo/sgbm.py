import cv2
import numpy as np
from stereo.stereoconfig import *
import cv2.ximgproc as ximgproc

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)


def getPoint3D_MODE3(img):
# 获取img的大小
    height, width = img.shape[:2]
    # 分割左右目图像
    left_image_path = img[:, :width // 2]
    right_image_path = img[:, width // 2:]

    # 将BGR格式转换成灰度图片，用于畸变矫正
    imgL = cv2.cvtColor(left_image_path, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(right_image_path, cv2.COLOR_BGR2GRAY)

    # 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片,输入图片要求为灰度图
    imgL = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    imgR = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    # 转换为opencv的BGR格式
    imgL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)
    radius = 2
    eps = 1000
    guided_filter = ximgproc.createGuidedFilter(guide=imgL, radius=radius, eps=eps)
    imgL = guided_filter.filter(imgL)
    imgR = guided_filter.filter(imgR)
    guided_filter = ximgproc.createGuidedFilter(guide=imgR, radius=radius, eps=eps)

    blockSize = 3
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=0,
                                numDisparities=128,
                                blockSize=blockSize,
                                P1=8 * img_channels * blockSize * blockSize,
                                P2=32 * img_channels * blockSize * blockSize,
                                disp12MaxDiff=-1,
                                preFilterCap=63,
                                uniquenessRatio=10,
                                speckleWindowSize=100,
                                speckleRange=1,
                                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    right_matcher = cv2.ximgproc.createRightMatcher(stereo)
    lmbda = 80000
    sigma = 2
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    # 计算视差图
    disparity_left = stereo.compute(imgL, imgR)
    disparity_right = right_matcher.compute(imgR, imgL)
    disparity_left = np.int16(disparity_left)
    disparity_right = np.int16(disparity_right)
    filteredImg = wls_filter.filter(disparity_left, imgL, imgR, disparity_right)

    mean = np.mean(filteredImg)
    std = np.std(filteredImg)
    filteredImg[filteredImg < mean - 2 * std] = mean
    filteredImg[filteredImg > mean + 2 * std] = mean

    # 归一化视差图
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    # 计算三维坐标数据值
    threeD = cv2.reprojectImageTo3D(filteredImg, Q, handleMissingValues=True)
    # 计算出的threeD，需要乘以16，才等于现实中的距离
    threeD = threeD * 16
    
    return threeD

def getPoint3D_MODE2(frame):
    height_0, width_0 = frame.shape[0:2]
    iml = frame[0:int(height_0), 0:int(width_0 / 2)]
    imr = frame[0:int(height_0), int(width_0 / 2):int(width_0)]
    # 消除图像畸变
    iml = undistortion(iml, left_camera_matrix, left_distortion)
    imr = undistortion(imr, right_camera_matrix, right_distortion)
    # 预处理图像: 彩色图->灰度图
    iml_, imr_ = preprocess(iml, imr)
    # 畸变校正和立体校正
    iml_rectified_l, imr_rectified_r = rectifyImage(iml_, imr_, left_map1, left_map2, right_map1, right_map2)
    # 使用SGBM算法进行立体匹配
    disp, _ = stereoMatchSGBM(iml_rectified_l, imr_rectified_r)
    # 使用Q矩阵将视差图转换为3D点云
    # cv2.reprojectImageTo3D函数将视差图转换为三维空间中的点
    # Q矩阵包含了相机的校正参数和立体匹配所需的变换矩阵
    # disp是之前计算得到的视差图
    # 函数返回一个3D点云
    # 返回的为目标中心点坐标的三维坐标
    threeD = cv2.reprojectImageTo3D(disp, Q)
    # 计算各值
    return threeD


def getPoint3D_MODE1(frame):
    # 获取图片的分辨率
    width = frame.shape[1]
    height = frame.shape[0]
    # 切割为左右两张图片
    frame1 = frame[0: height, 0: width // 2]
    frame2 = frame[0: height, width // 2: width]
    # 将BGR格式转换成灰度图片，用于畸变矫正
    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片,输入图片要求为灰度图
    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    # 转换为opencv的BGR格式
    imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

    # ------------------------------------SGBM算法----------------------------------------------------------
    #   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
    #   img_channels                BGR图像的颜色通道，img_channels=3，不可更改
    #   numDisparities              SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，如numDisparities
    #                               取16、32、48、64等
    #   mode                        sgbm算法选择模式，以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、
    #                               STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
    # ------------------------------------------------------------------------------------------------------
    blockSize = 3
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=1,
                                   numDisparities=64,
                                   blockSize=blockSize,
                                   P1=8 * img_channels * blockSize * blockSize,
                                   P2=32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=-1,
                                   preFilterCap=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=100,
                                   mode=cv2.STEREO_SGBM_MODE_HH)
    # 计算视差
    disparity = stereo.compute(img1_rectified, img2_rectified)

    # 归一化函数算法，生成深度图（灰度图）
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 生成深度图（颜色图）
    dis_color = disparity
    dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dis_color = cv2.applyColorMap(dis_color, 2)

    # 计算三维坐标数据值
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    # 计算出的threeD，需要乘以16，才等于现实中的距离
    threeD = threeD * 16
    
    return threeD



# 预处理图像的函数
def preprocess(img1, img2):
    # 如果图像是彩色的（三维数组），则将其转换为灰度图像
    # OpenCV默认加载的图像格式是BGR（蓝绿红），所以需要转换到灰度
    if(img1.ndim == 3):  # ndim属性用于检查数组的维度，3表示彩色图像
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if(img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # 对灰度图像应用直方图均衡化，以增强图像的对比度
    # 直方图均衡化是一种提高图像全局对比度的方法
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)
    # 返回预处理后的两个图像
    return img1, img2


# 消除图像畸变的函数
def undistortion(image, camera_matrix, dist_coeff):
    # 使用cv2.undistort函数来校正图像畸变
    # 该函数接受原始图像、相机内参矩阵和畸变系数作为输入
    # 它返回校正后的图像，其中畸变已经被消除
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)
    # 返回校正后的无畸变图像
    return undistortion_image


# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
# @param：config是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()
# 定义一个函数，用于获取立体相机校正变换所需的参数
def getRectifyTransform(height, width, config):
    # 从传入的config对象中获取左相机的内参矩阵
    left_K = config.cam_matrix_left
    # 从传入的config对象中获取右相机的内参矩阵
    right_K = config.cam_matrix_right
    # 从config对象中获取左相机的畸变系数
    left_distortion = config.distortion_l
    # 从config对象中获取右相机的畸变系数
    right_distortion = config.distortion_r
    # 获取右相机相对于左相机的旋转矩阵R
    R = config.R
    # 获取右相机相对于左相机的平移向量T
    T = config.T
    # 使用立体校正函数计算校正所需的变换矩阵和投影矩阵
    # alpha=0 表示不进行缩放，保持图像原始大小
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion, 
                                                    (width, height), R, T, alpha=0)
    # 为左相机图像创建一个用于畸变校正和变换的映射
    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    # 为右相机图像创建一个用于畸变校正和变换的映射
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)
    # 返回计算得到的映射和用于立体匹配的矩阵Q
    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
# 该函数接收两个图像和相应的映射矩阵，对图像进行畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    # 使用remap函数对第一个图像进行畸变校正
    # map1x和map1y是从getRectifyTransform函数中获取的映射矩阵，用于校正image1
    # cv2.INTER_AREA插值方法用于重采样，适合缩小图像尺寸
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    # 使用remap函数对第二个图像进行畸变校正
    # map2x和map2y是从getRectifyTransform函数中获取的映射矩阵，用于校正image2
    # 同样使用cv2.INTER_AREA插值方法
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
    # 返回校正后的两个图像
    return rectifyed_img1, rectifyed_img2

# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    return output

# 使用SGBM算法进行立体匹配的函数
def stereoMatchSGBM(left_image, right_image):
    # 检查图像是单通道（灰度图）还是多通道（彩色图）
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    # 设置SGBM算法的参数
    blockSize = 5
    paraml = {
        'minDisparity': 0,  # 最小视差值
        'numDisparities': 160,  # 视差搜索范围
        'blockSize': blockSize,  # 用于计算视差的块大小
        'P1': 8 * img_channels * blockSize ** 2,  # 用于视差块匹配的惩罚参数
        'P2': 32 * img_channels * blockSize ** 2,  # 用于视差块匹配的惩罚参数
        'disp12MaxDiff': 5,  # 左视图和右视图视差的最大差异
        'preFilterCap': 63,  # 预处理图像时使用的最大值
        'uniquenessRatio': 15,  # 独特性比率，用于中止搜索视差
        'speckleWindowSize': 100,  # 噪声过滤的窗口大小
        'speckleRange': 1,  # 噪声过滤的视差范围
        'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY  # 使用的SGBM模式
    }
    # 创建左视图的SGBM匹配器对象
    left_matcher = cv2.StereoSGBM_create(**paraml)
    # 创建右视图的SGBM匹配器对象，调整minDisparity参数
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    # 获取左视图图像的尺寸
    size = (left_image.shape[1], left_image.shape[0])
    # 对图像进行下采样以加速计算
    left_image_down = cv2.pyrDown(left_image)
    right_image_down = cv2.pyrDown(right_image)
    # 计算下采样因子
    factor = left_image.shape[1] / left_image_down.shape[1]

    # 使用SGBM匹配器计算左视图和右视图的视差图
    disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
    disparity_right_half = right_matcher.compute(right_image_down, left_image_down)

    # 将下采样后的视差图上采样到原始图像的尺寸
    disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
    disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)

    # 调整下采样图像的视差值以反映原始图像的视差
    disparity_left = factor * disparity_left
    disparity_right = factor * disparity_right

    # 将视差图的值转换为真实的视差值（SGBM算法结果乘以16）
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.

    # 返回计算得到的左右视图的真实视差图
    return trueDisp_left, trueDisp_right