# archivo: test_cams.py
import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # en Windows va bien CAP_DSHOW
    if cap.isOpened():
        print(f"Camara encontrada en indice {i}")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f'cam {i}', frame)
            cv2.waitKey(1000)  # mostrar 1 segundo
        cap.release()

cv2.destroyAllWindows()
