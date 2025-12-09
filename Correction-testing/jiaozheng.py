import cv2  #librería open cv 
import numpy as np

def undistort_and_remap(image, camera_matrix, distortion, R, P, size):
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion, R, P, size, cv2.CV_32FC1)
    rectified_image = cv2.remap(image, map1, map2, cv2.INTER_CUBIC)
    return rectified_image


def cat2images(limg, rimg):
    HEIGHT = limg.shape[0]
    WIDTH = limg.shape[1]
    imgcat = np.zeros((HEIGHT, WIDTH * 2 + 20, 3))
    imgcat[:, :WIDTH, :] = limg
    imgcat[:, -WIDTH:, :] = rimg
    for i in range(int(HEIGHT / 32)):
        imgcat[i * 32, :, :] = 255
    return imgcat

# 这里为输入整张图片, 用cv2库函数进行切割
img= cv2.imread("251.jpg")
left_image = img[0:720, 0:1280]
right_image = img[0:720, 1280:2560]

imgcat_source = cat2images(left_image, right_image)
HEIGHT = left_image.shape[0]
WIDTH = left_image.shape[1]


camera_matrix0 = np.array([
    [2036.5, 1.3, 583.4],
    [0, 2028.6, 449.5],
    [0, 0, 1]])

distortion0 = np.array([[-0.3863, 0.5091, -0.0011, 0.001, -2.6353]])

camera_matrix1 = np.array([
    [2036.5, 1.3, 583.4],
    [0, 2028.6, 449.5],
    [0, 0, 1]])

distortion1 = np.array([[-0.4256, 1.5578, -0.0023, 0.0005, -11.6757]])

R = np.array([
    [0.9999, 0.0012, 0.0164],
    [-0.0011, 1, -0.0019],
    [-0.0164, 0.0019, 0.9999]])

T = np.array([[-85.9802], [0.0723], [1.1451]])

(R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2) = \
    cv2.stereoRectify(camera_matrix0, distortion0, camera_matrix1, distortion1, np.array([WIDTH, HEIGHT]), R,
                      T)  # 计算旋转矩阵和投影矩阵
# 矫正
rect_left_image = undistort_and_remap(left_image, camera_matrix0, distortion0, R_l, P_l, (WIDTH, HEIGHT))
rect_right_image = undistort_and_remap(right_image, camera_matrix1, distortion1, R_r, P_r, (WIDTH, HEIGHT))

imgcat_out = cat2images(rect_left_image, rect_right_image)
cv2.imwrite('imgcat_out.jpg', imgcat_out)
print(f"Done! Saved imgcat_out.jpg")