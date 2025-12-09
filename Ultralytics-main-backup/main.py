import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from main_win.win import Ui_mainWindow
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import time
import json
import numpy as np
import cv2
from utils.CustomMessageBox import MessageBox
from utils.capnums import Camera
from dialog.rtsp_win import Window
from ultralytics import YOLO

"""
从stereo文件夹里的代码导入必要的函数
"""

from inference import *

class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)  # 发送图片信号
    send_statistic = pyqtSignal(dict)  # 发送统计信号
    send_msg = pyqtSignal(str)  # 发送msg到label控件
    send_fps = pyqtSignal(str)   # 实时统计fps的信号
    send_percent = pyqtSignal(int)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './yolov8n.pt'   # 一些类的属性  用于调参
        self.current_weight = './yolov5n.pt'
        self.source = ''
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.jump_out = False                   # jump out of the loop
        self.is_continue = True                 # continue/pause
        self.percent_length = 1000              # progress bar
        self.rate_check = True                  # Whether to enable delay
        self.rate = 100
        self.save_fold = './result'  # 结果保存的路径
        self.save = False
        self.image_extensions = ["jpg", "jpeg", "png", "gif", "bmp"]
        self.img = False
        # 在这里设置你的摄像头分辨率
        self.height = 720
        self.width = 2560
        """
        注意, 如果你是自组摄像头, 需要自己写一下双目画面拼接, 该程序的输入为双目拼接画面
        """

    def run(self):
        try:

            """
            1.该区域代码主要是用于判断输入源的类型, 是视频还是图片还是摄像头
            """
            # -----------------------------------------------------
            if self.source.isdigit():
                self.source = int(self.source)
            file_extension = str(self.source).lower().split(".")[-1]
            if file_extension in self.image_extensions:
                self.img = True
            file_name = os.path.basename(str(self.source))

            # -----------------------------------------------------
            """
            以下区域代码主要是用于打开输入资源, 初始化yolov8推理所需要的东西, 然后进行循环推理
            """
            # -----------------------------------------------------
            cap = cv2.VideoCapture(self.source)  # cv读取输入源
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)   # 设置画面的宽高
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            model = YOLO(self.weights)
            if not self.img:
                # 创建视频写入对象
                w = 1280
                h = 720
                if type(self.source) != int:
                    folder = './result/' + file_name
                else:
                    folder = './result/' + str(self.source) + '.avi'
                vid_writer = cv2.VideoWriter(folder, cv2.VideoWriter_fourcc(*'XVID'), 25, (w, h))
            # -----------------------------------------------------
            # 循环推理入口
            while True:
                t1 = time.time()
                # 判断是否需要跳出循环(前端终止循环按钮的接口)
                if self.jump_out:
                    self.send_msg.emit('finished')  # 向label控件发送完成的信号
                    vid_writer.release()
                    break
                # 是否继续推理, 用来控制开始和暂停推理
                if self.is_continue:
                    ret, frame = cap.read()
                    if ret:
                        annotated_frame = infer(model=model, frame=frame)
                    if self.img:
                        folder = "./result/" + str(file_name)
                        cv2.imwrite(folder, annotated_frame)
                    else:
                        vid_writer.write(annotated_frame)
                    t2 = time.time()
                    text = "fps:" + str(int(1 / (t2 - t1)))
                    self.send_img.emit(annotated_frame)
                    self.send_fps.emit(text)
                else:
                    self.is_continue = False
        except Exception as e:
            print(e)

"""
MainWindow类就是用于pyqt5界面生成的, 这个不需要太了解
"""

class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False
        # 设置窗口样式
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
                            | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.minButton.clicked.connect(self.showMinimized)  # 点击minButton时最小化窗口
        self.maxButton.clicked.connect(self.max_or_restore)  # 未点击最大化窗口时未标准窗口 否则未最大化窗口
        self.maxButton.animateClick(10)  # 最大化窗口时给10ms响应时间
        self.closeButton.clicked.connect(self.close)  # closebutton 关闭窗口
        self.qtimer = QTimer(self)  # 创建定时器
        self.qtimer.setSingleShot(True)  #
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())  # 定期执行statistic_label.clear 用于清除上一次的统计信息
        self.comboBox.clear()
        self.pt_list = os.listdir('./pt')  # pt_list 用于寻找pt文件夹下的.pt结尾的模型文件
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/'+x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)  # 向下拉框中添加模型选项
        self.qtimer_search = QTimer(self)   #  创建定时器
        self.qtimer_search.timeout.connect(lambda: self.search_pt())  # 定期扫描
        self.qtimer_search.start(2000)  # 2s一次
        self.det_thread = DetThread()  # 创建推理线程
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.det_thread.source = '0'
        self.det_thread.percent_length = self.progressBar.maximum()  # 以下send都是发送信号用
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))  # 信号与槽 用于显示推理画面
        self.det_thread.send_statistic.connect(self.show_statistic)  # 发送统计信息
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))  # 发送消息信号
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))  # 设置进度条的值
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))  # 发送fps的值
        self.fileButton.clicked.connect(self.open_file)  # 打开文件的功能函数
        self.cameraButton.clicked.connect(self.chose_cam)  # 选择摄像头
        self.rtspButton.clicked.connect(self.chose_rtsp)  # 选择rtsp视频流地址
        self.runButton.clicked.connect(self.run_or_continue)  # 开始 暂停推理
        self.stopButton.clicked.connect(self.stop)  # 停止推理
        self.comboBox.currentTextChanged.connect(self.change_model)  # 更换模型
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))  # spinbox 和 slider控件结合实现实时调参
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))

        self.saveCheckBox.clicked.connect(self.is_save)  # 是否保存推理文件
        self.load_setting()

    def search_pt(self):
        try:
            pt_list = os.listdir('./pt')
            pt_list = [file for file in pt_list if file.endswith('.pt')]
            pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

            if pt_list != self.pt_list:
                self.pt_list = pt_list
                self.comboBox.clear()
                self.comboBox.addItems(self.pt_list)
        except Exception as e:
            print(e)


    def is_save(self):
        try:
            if self.saveCheckBox.isChecked():
                self.det_thread.save_fold = './result'
            else:
                self.det_thread.save_fold = None
        except Exception as e:
            print(e)


    def chose_rtsp(self):
        try:
            self.rtsp_window = Window()
            config_file = 'config/ip.json'
            if not os.path.exists(config_file):
                ip = "rtsp://admin:admin888@192.168.1.67:555"
                new_config = {"ip": ip}
                new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(new_json)
            else:
                config = json.load(open(config_file, 'r', encoding='utf-8'))
                ip = config['ip']
            self.rtsp_window.rtspEdit.setText(ip)
            self.rtsp_window.show()
            self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))
        except Exception as e:
            print(e)
    def load_rtsp(self, ip):
        try:
            self.stop()
            self.det_thread.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.statistic_msg('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.statistic_msg('%s' % e)

    def chose_cam(self):    # 选择摄像头
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='Tips', text='正在查找可用摄像头', time=500, auto=True).exec_()
            # get the number of local cameras
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.cameraButton.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).x()
            y = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).y()
            y = y + self.cameraButton.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec_(pos)
            if action:
                self.det_thread.source = action.text()
                self.statistic_msg('Loading camera：{}'.format(action.text()))
        except Exception as e:
            self.statistic_msg('%s' % e)

    def load_setting(self):  # 加载设置的参数
        try:
            config_file = 'config/setting.json'
            if not os.path.exists(config_file):
                iou = 0.26
                conf = 0.33
                savecheck = 0
                new_config = {"iou": iou,
                              "conf": conf,
                              "savecheck": savecheck
                              }
                new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(new_json)
            else:
                config = json.load(open(config_file, 'r', encoding='utf-8'))
                if len(config) != 3:
                    iou = 0.26
                    conf = 0.33
                    savecheck = 0
                else:
                    iou = config['iou']
                    conf = config['conf']
                    savecheck = config['savecheck']
            self.confSpinBox.setValue(iou)
            self.iouSpinBox.setValue(conf)
            self.saveCheckBox.setCheckState(savecheck)
            self.is_save()
        except Exception as e:
            print(e)
    def change_val(self, x, flag):
        try:
            if flag == 'confSpinBox':
                self.confSlider.setValue(int(x*100))
            elif flag == 'confSlider':
                self.confSpinBox.setValue(x/100)
                self.det_thread.conf_thres = x/100
            elif flag == 'iouSpinBox':
                self.iouSlider.setValue(int(x*100))
            elif flag == 'iouSlider':
                self.iouSpinBox.setValue(x/100)
                self.det_thread.iou_thres = x/100
            else:
                pass
        except Exception as e:
            print(e)

    def statistic_msg(self, msg):
        try:
            self.statistic_label.setText(msg)
        except Exception as e:
            print(e)

    def show_msg(self, msg):
        try:
            self.runButton.setChecked(Qt.Unchecked)
            self.statistic_msg(msg)
            if msg == "Finished":
                self.saveCheckBox.setEnabled(True)
        except Exception as e:
            print(e)
    def change_model(self, x):
        try:
            self.model_type = self.comboBox.currentText()
            self.det_thread.weights = "./pt/%s" % self.model_type
            self.statistic_msg('Change model to %s' % x)
        except Exception as e:
            print(e)
    def open_file(self):
        try:
            config_file = 'config/fold.json'
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            open_fold = config['open_fold']
            if not os.path.exists(open_fold):
                open_fold = os.getcwd()
            name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                              "*.jpg *.png)")
            if name:
                self.det_thread.source = name
                self.statistic_msg('Loaded file：{}'.format(os.path.basename(name)))
                config['open_fold'] = os.path.dirname(name)
                config_json = json.dumps(config, ensure_ascii=False, indent=2)
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(config_json)
                self.stop()
        except Exception as e:
            print(e)
    def max_or_restore(self):
        try:
            if self.maxButton.isChecked():
                self.showMaximized()
            else:
                self.showNormal()
        except Exception as e:
            print(e)
    def run_or_continue(self):
        try:
            self.det_thread.jump_out = False
            if self.runButton.isChecked():
                self.saveCheckBox.setEnabled(False)
                self.det_thread.is_continue = True
                if not self.det_thread.isRunning():
                    self.det_thread.start()
                source = os.path.basename(self.det_thread.source)
                source = 'camera' if source.isnumeric() else source
                self.statistic_msg('Detecting >> model：{}，file：{}'.
                                   format(os.path.basename(self.det_thread.weights),
                                          source))
            else:
                self.det_thread.is_continue = False
                self.statistic_msg('Pause')
        except Exception as e:
            print(e)
    def stop(self):
        try:
            self.det_thread.jump_out = True
            self.saveCheckBox.setEnabled(True)
        except Exception as e:
            print(e)
    def mousePressEvent(self, event):
        try:
            self.m_Position = event.pos()
            if event.button() == Qt.LeftButton:
                if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                        0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                    self.m_flag = True
        except Exception as e :
            print(e)
    def mouseMoveEvent(self, QMouseEvent):
        try:
            if Qt.LeftButton and self.m_flag:
                self.move(QMouseEvent.globalPos() - self.m_Position)
        except Exception as e:
            print(e)
    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))
            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))
        except Exception as e:
            print(repr(e))

    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' '+str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.resultWidget.addItems(results)

        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        try:
            self.det_thread.jump_out = True
            config_file = 'config/setting.json'
            config = dict()
            config['iou'] = self.confSpinBox.value()
            config['conf'] = self.iouSpinBox.value()
            config['savecheck'] = self.saveCheckBox.checkState()
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
        except Exception as e:
            print(e)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.windowTitle = 'yolov5'
    myWin.show()
    sys.exit(app.exec_())
