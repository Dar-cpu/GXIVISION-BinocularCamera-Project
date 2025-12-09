import cv2
from stereo.sgbm import *
"""
推理函数func, 输入model(模型), 原始图片, 返回处理好的图像
"""

def infer(model, frame):
    height_0, width_0 = frame.shape[0:2]
    results = model(frame[0: height_0, 0: width_0 // 2])[0]
    points_3d = None
    if SGBM_MODE_ == 1:
        points_3d = getPoint3D_MODE1(frame)
    elif SGBM_MODE_ == 2:
        points_3d = getPoint3D_MODE2(frame)
    elif SGBM_MODE_ == 3:
        points_3d = getPoint3D_MODE3(frame)
    for xyxy, conf, cls in zip(results.boxes.xyxy.tolist(), results.boxes.conf.tolist(), results.boxes.cls.tolist()):
        x1 = int(xyxy[0])
        y1 = int(xyxy[1])
        x2 = int(xyxy[2])
        y2 = int(xyxy[3])
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)
        if x <= int(width_0 / 2):
            # 计算距离
            dis = ((points_3d[int(y), int(x), 0] ** 2 + points_3d[int(y), int(x), 1] ** 2 + points_3d[
                int(y), int(x), 2] ** 2) ** 0.5) / 10
            # x,y,z分别为相对左目相机的世界坐标
            text_x = "x:%.1fcm" % (points_3d[int(y), int(x), 0] / 10)
            text_y = "y:%.1fcm" % (points_3d[int(y), int(x), 1] / 10)
            text_z = "z:%.1fcm" % (points_3d[int(y), int(x), 2] / 10)
            text_dis = "dis:%.1fcm" % dis
            # 以下皆为绘制距离信息和目标框信息
            cv2.rectangle(frame, (int(xyxy[0] + (xyxy[2] - xyxy[0])), int(xyxy[1])),
                        (int(xyxy[0] + (xyxy[2] - xyxy[0])) + 5 + 220, int(xyxy[1] + 150)), (0, 0, 0),
                        -1)
            cv2.putText(frame, text_x, (int(xyxy[0] + (xyxy[2] - xyxy[0]) + 5), int(xyxy[1] + 30)), cv2.FONT_ITALIC,
                        1.2, (255, 255, 255), 3)
            cv2.putText(frame, text_y, (int(xyxy[0] + (xyxy[2] - xyxy[0]) + 5), int(xyxy[1] + 65)), cv2.FONT_ITALIC,
                        1.2, (255, 255, 255), 3)
            cv2.putText(frame, text_z, (int(xyxy[0]+(xyxy[2]-xyxy[0])+5), int(xyxy[1]+100)), cv2.FONT_ITALIC,
                        1.2, (255, 255, 255), 3)
            cv2.putText(frame, text_dis, (int(xyxy[0] + (xyxy[2] - xyxy[0]) + 5), int(xyxy[1] + 145)),
                        cv2.FONT_ITALIC, 1.2, (255, 255, 255), 3)
            # 随机颜色
            color = random_color(int(cls))           
            # 绘制坐标框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2, lineType=cv2.LINE_AA)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            w, h = cv2.getTextSize(label, 0, 1, 2)[0]
            cv2.rectangle(frame, (x1 - 3, y1 - 24), (x1 + w - 50, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 0, 0.6, (0, 0, 0), 2, 1) 
    # 设置为仅显示左目相机内容
    frame = frame[0:height_0, 0:int(width_0 / 2)]
    return frame

"""
绘图用函数
"""
def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    r, g, b = 0, 0, 0

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return int(b * 255), int(g * 255), int(r * 255)

def random_color(id):
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)