import numpy as np

# SGBM模式
"""
MODE1: 默认模式, 使用原始SGBM算法,  匹配速度适中
MODE2: 下采样模式, 对图像进行了下采样处理, 匹配速度较快, 丢失一定精度
MODE3: WLS滤波处理模式, 使用WLS滤波协助SGBM匹配, 速度较慢, 丢失一定精度, 但解决了深度图稀碎问题
"""
SGBM_MODE_ = 1


# -----------------------------------双目相机的基本参数---------------------------------------------------------
#   left_camera_matrix          左相机的内参矩阵
#   right_camera_matrix         右相机的内参矩阵
#
#   left_distortion             左相机的畸变系数    格式(K1,K2,P1,P2,0)
#   right_distortion            右相机的畸变系数
# -------------------------------------------------------------------------------------------------------------
# 左镜头的内参，如焦距
left_camera_matrix = np.array([
    [1.1575, 0, 0],
    [0.0001, 1.1579, 0],
    [0.5916, 0.3404, 0.0010]  # datos tomados de matlab, luego de la calibracion 
])


right_camera_matrix = np.array([
    [1.1510, 0, 0],
    [0.0009, 1.1510, 0],
    [0.6200, 0.3568, 0.0010]    # datos tomados de matlab, luego de la calibracion 
])


# 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变
left_distortion = np.array([[0.0397, 0.1860, -0.1601, -0.3299, -1.1144]])     # datos tomados de matlab, luego de la calibracion 
right_distortion = np.array([[0.0515 , 0.0898, -0.0013, -0.0026, -0.9678 ]])     # datos tomados de matlab, luego de la calibracion 
 
# 旋转矩阵
R = np.array([
    [1.0000, 0.0002, 0.0096],
    [-0.001, 1.0000, -0.0076],
    [-0.0096, 0.0076, 0.9999]  # datos tomados de matlab, luego de la calibracion 
])

# 平移矩阵
T = np.array([[-87.9492], [0.2741], [-1.8581]])     # datos tomados de matlab, luego de la calibracion 

size = (1280, 720)