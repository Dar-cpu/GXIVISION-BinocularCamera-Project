import cv2
import os
import time

# 设置拍照间隔时间（单位：秒）
interval = 4

# 设置保存目录
left_dir = 'left'
right_dir = 'right'

# 创建保存目录（如果不存在）
os.makedirs(left_dir, exist_ok=True)
os.makedirs(right_dir, exist_ok=True)


# 设置拍照函数
def cap():
    # 设置视频流或者摄像头 
    cap = cv2.VideoCapture(2)  #Camara 2 en caso de tener camara principal 
    # 设置画面的宽高，这里设置为2560x720, 根据自己双目相机的分辨率来
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    count = 1  # 初始化图片计数器
    # 获取当前时间戳
    start_time = time.time()
    while True:
        
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # 分割为左右目图片
        imgL = frame[:, :1280]  # 左半部分
        imgR = frame[:, 1280:]  # 右半部分
    
        # 显示左右目图片
        cv2.imshow('left', imgL)
        cv2.imshow('right', imgR)
        
        # 获取当前时间戳
        current_time = time.time()
        
        # 每隔interval秒保存一次图片
        if current_time - start_time >= interval:
            left_filename = os.path.join(left_dir, f'left_{count}.jpg')
            right_filename = os.path.join(right_dir, f'right_{count}.jpg')
            
            cv2.imwrite(left_filename, imgL)
            cv2.imwrite(right_filename, imgR)
            print(f'Saved {left_filename} and {right_filename}')
            
            # 更新开始时间
            start_time = current_time
            count += 1
        
        # 等待1毫秒
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cap()
